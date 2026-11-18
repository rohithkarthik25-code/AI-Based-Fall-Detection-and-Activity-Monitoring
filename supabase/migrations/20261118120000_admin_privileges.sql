-- =============================================================================
-- Migration: 20261118120000_admin_privileges.sql
-- Project Reference: qbzuwqanxnkpeuioiiao
-- Depends on: 20261117183000_auth_roles_and_email_alerts.sql
--
-- Adds real admin powers. Every power is enforced in the database (RLS +
-- SECURITY DEFINER RPCs that re-check is_admin()), never only in the UI.
--
--   1. Support tickets are no longer world-readable (privacy fix).
--   2. Admin activity log: append-only, admin-read-only.
--   3. Role management: admin_set_user_role() with last-admin protection.
--   4. Block / unblock users: admin_set_user_blocked(); blocked users lose
--      all data access via RLS (fall_events, alert settings, uploads, tickets).
--   5. Soft-delete false alarms: admin_delete_fall_event() / admin_restore_fall_event().
--   6. Support ticket inbox: admin_update_support_ticket().
--   7. Incident resolve / reopen is recorded in the activity log automatically.
--
-- Concurrency: role/block changes take one transaction-scoped advisory lock and
-- re-check is_admin() after acquiring it, so two admins cannot demote each other
-- at the same moment and leave the system with zero admins.
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 1. Blocking columns on user_roles
-- -----------------------------------------------------------------------------
ALTER TABLE public.user_roles
    ADD COLUMN IF NOT EXISTS blocked_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS blocked_by UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS blocked_reason TEXT,
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT now();

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'user_roles_blocked_reason_len') THEN
        ALTER TABLE public.user_roles
            ADD CONSTRAINT user_roles_blocked_reason_len
            CHECK (blocked_reason IS NULL OR length(blocked_reason) <= 300);
    END IF;
END $$;

-- -----------------------------------------------------------------------------
-- 2. Helper functions
-- -----------------------------------------------------------------------------
-- True when the current caller's account is blocked.
CREATE OR REPLACE FUNCTION public.is_blocked()
RETURNS BOOLEAN
LANGUAGE sql
SECURITY DEFINER
STABLE
SET search_path = public, pg_temp
AS $$
    SELECT EXISTS (
        SELECT 1 FROM public.user_roles
        WHERE user_id = auth.uid()
          AND blocked_at IS NOT NULL
    );
$$;

REVOKE EXECUTE ON FUNCTION public.is_blocked() FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.is_blocked() TO authenticated, service_role;

-- A blocked account is never treated as admin (defence in depth; admins cannot be blocked).
CREATE OR REPLACE FUNCTION public.is_admin()
RETURNS BOOLEAN
LANGUAGE sql
SECURITY DEFINER
STABLE
SET search_path = public, pg_temp
AS $$
    SELECT EXISTS (
        SELECT 1 FROM public.user_roles
        WHERE user_id = auth.uid()
          AND role = 'admin'
          AND blocked_at IS NULL
    );
$$;

REVOKE EXECUTE ON FUNCTION public.is_admin() FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.is_admin() TO authenticated, service_role;

-- -----------------------------------------------------------------------------
-- 3. Admin activity log (append-only)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.admin_audit_log (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    actor_id UUID,              -- no FK: the log must survive account deletion
    actor_email TEXT,
    action TEXT NOT NULL,
    target_type TEXT NOT NULL,
    target_id TEXT,
    details JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_admin_audit_log_created ON public.admin_audit_log (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_admin_audit_log_target ON public.admin_audit_log (target_type, target_id);

ALTER TABLE public.admin_audit_log ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Admins read activity log" ON public.admin_audit_log;
CREATE POLICY "Admins read activity log"
    ON public.admin_audit_log
    FOR SELECT
    TO authenticated
    USING (public.is_admin());

-- Clients can only read (and only admins, via RLS). Writes go through write_audit_log().
REVOKE INSERT, UPDATE, DELETE, TRUNCATE ON public.admin_audit_log FROM PUBLIC, anon, authenticated;

CREATE OR REPLACE FUNCTION public.prevent_audit_log_changes()
RETURNS TRIGGER
LANGUAGE plpgsql
SET search_path = public, pg_temp
AS $$
BEGIN
    RAISE EXCEPTION 'admin_audit_log is append-only: entries cannot be changed or deleted.';
END;
$$;

REVOKE EXECUTE ON FUNCTION public.prevent_audit_log_changes() FROM PUBLIC, anon, authenticated;

DROP TRIGGER IF EXISTS trg_audit_log_no_update_delete ON public.admin_audit_log;
CREATE TRIGGER trg_audit_log_no_update_delete
    BEFORE UPDATE OR DELETE ON public.admin_audit_log
    FOR EACH ROW
    EXECUTE FUNCTION public.prevent_audit_log_changes();

DROP TRIGGER IF EXISTS trg_audit_log_no_truncate ON public.admin_audit_log;
CREATE TRIGGER trg_audit_log_no_truncate
    BEFORE TRUNCATE ON public.admin_audit_log
    FOR EACH STATEMENT
    EXECUTE FUNCTION public.prevent_audit_log_changes();

-- Internal writer. Not callable by clients; only by the SECURITY DEFINER functions below.
CREATE OR REPLACE FUNCTION public.write_audit_log(
    p_action TEXT,
    p_target_type TEXT,
    p_target_id TEXT,
    p_details JSONB DEFAULT '{}'::jsonb
)
RETURNS VOID
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
DECLARE
    v_email TEXT;
BEGIN
    SELECT email INTO v_email FROM auth.users WHERE id = auth.uid();
    INSERT INTO public.admin_audit_log (actor_id, actor_email, action, target_type, target_id, details)
    VALUES (auth.uid(), v_email, p_action, p_target_type, p_target_id, COALESCE(p_details, '{}'::jsonb));
END;
$$;

REVOKE EXECUTE ON FUNCTION public.write_audit_log(TEXT, TEXT, TEXT, JSONB) FROM PUBLIC, anon, authenticated;

-- -----------------------------------------------------------------------------
-- 4. Role management
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION public.admin_set_user_role(p_user_id UUID, p_role TEXT)
RETURNS VOID
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
DECLARE
    v_old_role TEXT;
    v_blocked TIMESTAMPTZ;
    v_active_admins INT;
BEGIN
    -- Serialise all role/block changes, then check permissions on fresh data.
    PERFORM pg_advisory_xact_lock(hashtext('fallguard_admin_role_changes'));

    IF NOT public.is_admin() THEN
        RAISE EXCEPTION 'Permission denied: admin only.' USING ERRCODE = '42501';
    END IF;
    IF p_role IS NULL OR p_role NOT IN ('admin', 'user') THEN
        RAISE EXCEPTION 'Invalid role: must be admin or user.';
    END IF;
    IF p_user_id = auth.uid() THEN
        RAISE EXCEPTION 'You cannot change your own role.';
    END IF;

    SELECT role, blocked_at INTO v_old_role, v_blocked
    FROM public.user_roles WHERE user_id = p_user_id FOR UPDATE;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'User not found.';
    END IF;
    IF v_old_role = p_role THEN
        RETURN;
    END IF;
    IF p_role = 'admin' AND v_blocked IS NOT NULL THEN
        RAISE EXCEPTION 'Unblock this user before making them an admin.';
    END IF;

    IF v_old_role = 'admin' THEN
        SELECT count(*) INTO v_active_admins
        FROM public.user_roles WHERE role = 'admin' AND blocked_at IS NULL;
        IF v_active_admins <= 1 THEN
            RAISE EXCEPTION 'Cannot remove the last active admin.';
        END IF;
    END IF;

    UPDATE public.user_roles
    SET role = p_role, updated_at = now()
    WHERE user_id = p_user_id;

    PERFORM public.write_audit_log(
        CASE WHEN p_role = 'admin' THEN 'ROLE_PROMOTED' ELSE 'ROLE_DEMOTED' END,
        'user', p_user_id::text,
        jsonb_build_object('from', v_old_role, 'to', p_role)
    );
END;
$$;

REVOKE EXECUTE ON FUNCTION public.admin_set_user_role(UUID, TEXT) FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.admin_set_user_role(UUID, TEXT) TO authenticated;

-- -----------------------------------------------------------------------------
-- 5. Block / unblock users
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION public.admin_set_user_blocked(p_user_id UUID, p_blocked BOOLEAN, p_reason TEXT DEFAULT NULL)
RETURNS VOID
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
DECLARE
    v_role TEXT;
    v_blocked_at TIMESTAMPTZ;
    v_reason TEXT := NULLIF(trim(COALESCE(p_reason, '')), '');
BEGIN
    PERFORM pg_advisory_xact_lock(hashtext('fallguard_admin_role_changes'));

    IF NOT public.is_admin() THEN
        RAISE EXCEPTION 'Permission denied: admin only.' USING ERRCODE = '42501';
    END IF;
    IF p_blocked IS NULL THEN
        RAISE EXCEPTION 'p_blocked must be true or false.';
    END IF;
    IF p_user_id = auth.uid() THEN
        RAISE EXCEPTION 'You cannot block or unblock yourself.';
    END IF;
    IF v_reason IS NOT NULL AND length(v_reason) > 300 THEN
        RAISE EXCEPTION 'Reason is too long (maximum 300 characters).';
    END IF;

    SELECT role, blocked_at INTO v_role, v_blocked_at
    FROM public.user_roles WHERE user_id = p_user_id FOR UPDATE;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'User not found.';
    END IF;

    IF p_blocked THEN
        IF v_role = 'admin' THEN
            RAISE EXCEPTION 'Admins cannot be blocked. Remove their admin role first.';
        END IF;
        IF v_reason IS NULL THEN
            RAISE EXCEPTION 'A reason is required to block a user.';
        END IF;
        IF v_blocked_at IS NOT NULL THEN
            RETURN;
        END IF;
        UPDATE public.user_roles
        SET blocked_at = now(), blocked_by = auth.uid(), blocked_reason = v_reason, updated_at = now()
        WHERE user_id = p_user_id;
        PERFORM public.write_audit_log('USER_BLOCKED', 'user', p_user_id::text,
                                       jsonb_build_object('reason', v_reason));
    ELSE
        IF v_blocked_at IS NULL THEN
            RETURN;
        END IF;
        UPDATE public.user_roles
        SET blocked_at = NULL, blocked_by = NULL, blocked_reason = NULL, updated_at = now()
        WHERE user_id = p_user_id;
        PERFORM public.write_audit_log('USER_UNBLOCKED', 'user', p_user_id::text,
                                       jsonb_build_object('reason', v_reason));
    END IF;
END;
$$;

REVOKE EXECUTE ON FUNCTION public.admin_set_user_blocked(UUID, BOOLEAN, TEXT) FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.admin_set_user_blocked(UUID, BOOLEAN, TEXT) TO authenticated;

-- -----------------------------------------------------------------------------
-- 6. Soft delete of fall_events (false alarms)
-- -----------------------------------------------------------------------------
ALTER TABLE public.fall_events
    ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS deleted_by UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS delete_reason TEXT;

CREATE INDEX IF NOT EXISTS idx_fall_events_deleted_at ON public.fall_events (deleted_at);

-- Column lock: unchanged for clients (only resolved, resolved_at, notes), but now
--  * lets trusted SECURITY DEFINER functions (running as the owner) through, and
--  * freezes deleted incidents for normal callers.
CREATE OR REPLACE FUNCTION public.enforce_fall_events_user_column_lock()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = public, pg_temp
AS $$
BEGIN
    -- service_role, direct SQL maintenance, and the admin_* RPCs (which run as the owner)
    IF (auth.role() = 'service_role'
        OR (auth.role() IS NULL AND session_user IN ('postgres', 'supabase_admin'))
        OR current_user IN ('postgres', 'supabase_admin')) THEN
        RETURN NEW;
    END IF;

    IF OLD.deleted_at IS NOT NULL THEN
        RAISE EXCEPTION 'Permission denied: this incident has been deleted.';
    END IF;

    IF (to_jsonb(NEW) - '{resolved, resolved_at, notes}'::text[] IS DISTINCT FROM
        to_jsonb(OLD) - '{resolved, resolved_at, notes}'::text[]) THEN
        RAISE EXCEPTION 'Permission denied: Only resolved, resolved_at, and notes may be updated.';
    END IF;

    RETURN NEW;
END;
$$;

REVOKE EXECUTE ON FUNCTION public.enforce_fall_events_user_column_lock() FROM PUBLIC, anon;

CREATE OR REPLACE FUNCTION public.admin_delete_fall_event(p_event_id UUID, p_reason TEXT)
RETURNS VOID
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
DECLARE
    v_reason TEXT := NULLIF(trim(COALESCE(p_reason, '')), '');
    v_row public.fall_events%ROWTYPE;
BEGIN
    IF NOT public.is_admin() THEN
        RAISE EXCEPTION 'Permission denied: admin only.' USING ERRCODE = '42501';
    END IF;
    IF v_reason IS NULL THEN
        RAISE EXCEPTION 'A reason is required to delete an incident.';
    END IF;
    IF length(v_reason) > 300 THEN
        RAISE EXCEPTION 'Reason is too long (maximum 300 characters).';
    END IF;

    UPDATE public.fall_events
    SET deleted_at = now(), deleted_by = auth.uid(), delete_reason = v_reason
    WHERE id = p_event_id AND deleted_at IS NULL
    RETURNING * INTO v_row;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'Incident not found or already deleted.';
    END IF;

    PERFORM public.write_audit_log('INCIDENT_DELETED', 'incident', p_event_id::text,
        jsonb_build_object('reason', v_reason, 'severity', v_row.severity,
                           'person_id', v_row.person_id, 'owner_id', v_row.user_id,
                           'incident_time', v_row.created_at));
END;
$$;

REVOKE EXECUTE ON FUNCTION public.admin_delete_fall_event(UUID, TEXT) FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.admin_delete_fall_event(UUID, TEXT) TO authenticated;

CREATE OR REPLACE FUNCTION public.admin_restore_fall_event(p_event_id UUID)
RETURNS VOID
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
DECLARE
    v_old_reason TEXT;
BEGIN
    IF NOT public.is_admin() THEN
        RAISE EXCEPTION 'Permission denied: admin only.' USING ERRCODE = '42501';
    END IF;

    SELECT delete_reason INTO v_old_reason
    FROM public.fall_events WHERE id = p_event_id AND deleted_at IS NOT NULL FOR UPDATE;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'Incident not found or not deleted.';
    END IF;

    UPDATE public.fall_events
    SET deleted_at = NULL, deleted_by = NULL, delete_reason = NULL
    WHERE id = p_event_id;

    PERFORM public.write_audit_log('INCIDENT_RESTORED', 'incident', p_event_id::text,
        jsonb_build_object('previous_reason', v_old_reason));
END;
$$;

REVOKE EXECUTE ON FUNCTION public.admin_restore_fall_event(UUID) FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.admin_restore_fall_event(UUID) TO authenticated;

-- Record every resolve / reopen (by users and admins) in the activity log.
CREATE OR REPLACE FUNCTION public.log_fall_event_resolution()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
BEGIN
    IF auth.uid() IS NOT NULL AND OLD.resolved IS DISTINCT FROM NEW.resolved THEN
        PERFORM public.write_audit_log(
            CASE WHEN NEW.resolved THEN 'INCIDENT_RESOLVED' ELSE 'INCIDENT_REOPENED' END,
            'incident', NEW.id::text,
            jsonb_build_object('severity', NEW.severity, 'owner_id', NEW.user_id)
        );
    END IF;
    RETURN NEW;
END;
$$;

REVOKE EXECUTE ON FUNCTION public.log_fall_event_resolution() FROM PUBLIC, anon, authenticated;

DROP TRIGGER IF EXISTS trg_log_fall_event_resolution ON public.fall_events;
CREATE TRIGGER trg_log_fall_event_resolution
    AFTER UPDATE OF resolved ON public.fall_events
    FOR EACH ROW
    EXECUTE FUNCTION public.log_fall_event_resolution();

-- -----------------------------------------------------------------------------
-- 7. fall_events RLS: hide deleted rows from users, cut off blocked users
-- -----------------------------------------------------------------------------
DROP POLICY IF EXISTS "Admins see all; users see own fall events" ON public.fall_events;
CREATE POLICY "Admins see all; users see own fall events"
    ON public.fall_events
    FOR SELECT
    TO authenticated, service_role
    USING (
        auth.role() = 'service_role'
        OR public.is_admin()
        OR (auth.uid() = user_id AND deleted_at IS NULL AND NOT public.is_blocked())
    );

DROP POLICY IF EXISTS "Authenticated users insert own events; admins and service role insert any" ON public.fall_events;
CREATE POLICY "Authenticated users insert own events; admins and service role insert any"
    ON public.fall_events
    FOR INSERT
    TO authenticated, service_role
    WITH CHECK (
        auth.role() = 'service_role'
        OR public.is_admin()
        OR (auth.uid() IS NOT NULL AND auth.uid() = user_id AND NOT public.is_blocked())
    );

DROP POLICY IF EXISTS "Admins update all; users update own fall events" ON public.fall_events;
CREATE POLICY "Admins update all; users update own fall events"
    ON public.fall_events
    FOR UPDATE
    TO authenticated, service_role
    USING (
        auth.role() = 'service_role'
        OR public.is_admin()
        OR (auth.uid() IS NOT NULL AND auth.uid() = user_id AND deleted_at IS NULL AND NOT public.is_blocked())
    )
    WITH CHECK (
        auth.role() = 'service_role'
        OR public.is_admin()
        OR (auth.uid() IS NOT NULL AND auth.uid() = user_id AND NOT public.is_blocked())
    );

-- Blocked users cannot change their alert settings either.
DROP POLICY IF EXISTS "Users update own alert settings" ON public.user_alert_settings;
CREATE POLICY "Users update own alert settings"
    ON public.user_alert_settings
    FOR UPDATE
    TO authenticated
    USING (auth.uid() = user_id AND NOT public.is_blocked())
    WITH CHECK (auth.uid() = user_id AND NOT public.is_blocked());

-- Blocked users cannot upload snapshots.
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_tables WHERE schemaname = 'storage' AND tablename = 'objects') THEN
        DROP POLICY IF EXISTS "Allow authenticated and service_role upload" ON storage.objects;
        CREATE POLICY "Allow authenticated and service_role upload"
            ON storage.objects
            FOR INSERT
            TO authenticated, service_role
            WITH CHECK (
                bucket_id = 'fall-snapshots'
                AND name ~ '^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\.jpg$'
                AND (auth.role() = 'service_role' OR NOT public.is_blocked())
            );
    END IF;
END $$;

-- -----------------------------------------------------------------------------
-- 8. Support tickets: privacy fix + admin inbox
-- -----------------------------------------------------------------------------
ALTER TABLE public.support_requests
    ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS admin_notes TEXT,
    ADD COLUMN IF NOT EXISTS handled_by UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS resolved_at TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS idx_support_requests_user_id ON public.support_requests (user_id);

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'support_requests_admin_notes_len') THEN
        ALTER TABLE public.support_requests
            ADD CONSTRAINT support_requests_admin_notes_len
            CHECK (admin_notes IS NULL OR length(admin_notes) <= 2000);
    END IF;
END $$;

-- Remove the old world-readable / anonymous policies.
DROP POLICY IF EXISTS "Allow public read support requests" ON public.support_requests;
DROP POLICY IF EXISTS "Allow public insert support requests" ON public.support_requests;
DROP POLICY IF EXISTS "Signed-in users submit tickets" ON public.support_requests;
DROP POLICY IF EXISTS "Users read own tickets; admins read all" ON public.support_requests;

CREATE POLICY "Signed-in users submit tickets"
    ON public.support_requests
    FOR INSERT
    TO authenticated
    WITH CHECK (NOT public.is_blocked());

CREATE POLICY "Users read own tickets; admins read all"
    ON public.support_requests
    FOR SELECT
    TO authenticated
    USING (public.is_admin() OR (user_id = auth.uid() AND user_id IS NOT NULL));

-- No UPDATE/DELETE policies: tickets are changed only through admin_update_support_ticket().

-- Server-controlled fields on insert; basic size limits on user input.
CREATE OR REPLACE FUNCTION public.enforce_support_request_insert()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = public, pg_temp
AS $$
BEGIN
    IF (auth.role() = 'service_role'
        OR (auth.role() IS NULL AND session_user IN ('postgres', 'supabase_admin'))) THEN
        RETURN NEW;
    END IF;

    IF length(COALESCE(NEW.full_name, '')) > 120
       OR length(COALESCE(NEW.email, '')) > 254
       OR length(COALESCE(NEW.subject, '')) > 200
       OR length(COALESCE(NEW.description, '')) > 5000
       OR length(COALESCE(NEW.attachment_url, '')) > 500 THEN
        RAISE EXCEPTION 'Support request fields exceed the allowed length.';
    END IF;

    NEW.user_id := auth.uid();
    NEW.created_at := now();
    NEW.status := 'OPEN';
    NEW.admin_notes := NULL;
    NEW.handled_by := NULL;
    NEW.updated_at := NULL;
    NEW.resolved_at := NULL;
    RETURN NEW;
END;
$$;

REVOKE EXECUTE ON FUNCTION public.enforce_support_request_insert() FROM PUBLIC, anon;

DROP TRIGGER IF EXISTS trg_enforce_support_request_insert ON public.support_requests;
CREATE TRIGGER trg_enforce_support_request_insert
    BEFORE INSERT ON public.support_requests
    FOR EACH ROW
    EXECUTE FUNCTION public.enforce_support_request_insert();

CREATE OR REPLACE FUNCTION public.admin_update_support_ticket(
    p_ticket_id UUID,
    p_status TEXT,
    p_priority TEXT,
    p_admin_notes TEXT
)
RETURNS VOID
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
DECLARE
    v_old_status TEXT;
    v_old_priority TEXT;
    v_notes TEXT := NULLIF(trim(COALESCE(p_admin_notes, '')), '');
BEGIN
    IF NOT public.is_admin() THEN
        RAISE EXCEPTION 'Permission denied: admin only.' USING ERRCODE = '42501';
    END IF;
    IF p_status IS NULL OR p_status NOT IN ('OPEN', 'IN_REVIEW', 'RESOLVED', 'CLOSED') THEN
        RAISE EXCEPTION 'Invalid status.';
    END IF;
    IF p_priority IS NULL OR p_priority NOT IN ('LOW', 'NORMAL', 'HIGH', 'CRITICAL') THEN
        RAISE EXCEPTION 'Invalid priority.';
    END IF;
    IF v_notes IS NOT NULL AND length(v_notes) > 2000 THEN
        RAISE EXCEPTION 'Notes are too long (maximum 2000 characters).';
    END IF;

    SELECT status, priority INTO v_old_status, v_old_priority
    FROM public.support_requests WHERE id = p_ticket_id FOR UPDATE;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'Ticket not found.';
    END IF;

    UPDATE public.support_requests
    SET status = p_status,
        priority = p_priority,
        admin_notes = v_notes,
        handled_by = auth.uid(),
        updated_at = now(),
        resolved_at = CASE
            WHEN p_status IN ('RESOLVED', 'CLOSED') THEN COALESCE(resolved_at, now())
            ELSE NULL
        END
    WHERE id = p_ticket_id;

    PERFORM public.write_audit_log('TICKET_UPDATED', 'ticket', p_ticket_id::text,
        jsonb_build_object('from_status', v_old_status, 'to_status', p_status,
                           'from_priority', v_old_priority, 'to_priority', p_priority));
END;
$$;

REVOKE EXECUTE ON FUNCTION public.admin_update_support_ticket(UUID, TEXT, TEXT, TEXT) FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.admin_update_support_ticket(UUID, TEXT, TEXT, TEXT) TO authenticated;
