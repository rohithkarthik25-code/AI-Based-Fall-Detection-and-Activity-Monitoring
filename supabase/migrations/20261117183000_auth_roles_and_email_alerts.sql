-- =============================================================================
-- Migration: 20261117183000_auth_roles_and_email_alerts.sql
-- Project Reference: qbzuwqanxnkpeuioiiao
-- Description:
--   1. Creates public.user_roles (admin, user) with RLS and helper is_admin().
--   2. Creates public.user_alert_settings with RLS and JSONB-strip column lock.
--   3. Adds user_id, email claim, attempt count, and status columns to fall_events.
--   4. Replaces open public policies on fall_events with strict per-user/admin RLS.
--   5. Hardens fall_events INSERT with user_id verification, input range validation,
--      forced server-controlled ID & created_at, per-user advisory lock, and dual rate limits.
--   6. Enforces allowlist column locks on fall_events UPDATE via JSONB key-stripping,
--      with explicit bypass for direct SQL maintenance users (postgres, supabase_admin, service_role).
--   7. Configures safe, non-blocking auth.users trigger that grants admin ONLY to
--      verified Google OAuth accounts matching 'rohithkarthik25@gmail.com'.
--   8. Idempotently backfills existing auth.users without overwriting roles.
--   9. Secures storage.objects for fall-snapshots bucket: enforces 2MB limit, image/jpeg,
--      UUID.jpg filename regex, removes public SELECT enumeration, allows auth/service upload.
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 1. Table: public.user_roles
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.user_roles (
    user_id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('admin', 'user')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE public.user_roles ENABLE ROW LEVEL SECURITY;

-- Helper function to check if current caller is an admin (takes no params; checks only auth.uid())
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
    );
$$;

-- Function permission lockdown
REVOKE EXECUTE ON FUNCTION public.is_admin() FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.is_admin() TO authenticated, service_role;

-- RLS: Users can view their own role; Admins can view all roles.
-- No client INSERT/UPDATE/DELETE allowed (only service_role and triggers can write).
DROP POLICY IF EXISTS "Users and admins can view roles" ON public.user_roles;
DROP POLICY IF EXISTS "Users view own role; admins view all" ON public.user_roles;
CREATE POLICY "Users view own role; admins view all"
    ON public.user_roles
    FOR SELECT
    TO authenticated
    USING (
        auth.uid() = user_id OR public.is_admin()
    );

-- -----------------------------------------------------------------------------
-- 2. Table: public.user_alert_settings
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.user_alert_settings (
    user_id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    email TEXT NOT NULL,
    email_alerts_enabled BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE public.user_alert_settings ENABLE ROW LEVEL SECURITY;

-- RLS: Users can view their own settings; Admins can view all settings.
DROP POLICY IF EXISTS "Users and admins can view alert settings" ON public.user_alert_settings;
DROP POLICY IF EXISTS "Users view own settings; admins view all" ON public.user_alert_settings;
CREATE POLICY "Users view own settings; admins view all"
    ON public.user_alert_settings
    FOR SELECT
    TO authenticated
    USING (
        auth.uid() = user_id OR public.is_admin()
    );

-- RLS: Users can update only their own row.
DROP POLICY IF EXISTS "Users can update own alert settings" ON public.user_alert_settings;
DROP POLICY IF EXISTS "Users update own alert settings" ON public.user_alert_settings;
CREATE POLICY "Users update own alert settings"
    ON public.user_alert_settings
    FOR UPDATE
    TO authenticated
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

-- Column-level protection via JSONB-stripping:
-- Users may only modify email_alerts_enabled and updated_at.
-- Modifications to user_id, email, created_at, or any other column are strictly rejected.
-- Maintenance users (postgres, supabase_admin, service_role) bypass this restriction.
CREATE OR REPLACE FUNCTION public.enforce_user_alert_settings_column_lock()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = public, pg_temp
AS $$
BEGIN
    -- Direct SQL maintenance users (postgres, supabase_admin in SQL Editor) and service_role bypass column locks
    IF (auth.role() = 'service_role' OR (auth.role() IS NULL AND session_user IN ('postgres', 'supabase_admin'))) THEN
        NEW.updated_at = now();
        RETURN NEW;
    END IF;

    -- Compare records with allowlisted fields removed
    IF (to_jsonb(NEW) - '{email_alerts_enabled, updated_at}'::text[] IS DISTINCT FROM
        to_jsonb(OLD) - '{email_alerts_enabled, updated_at}'::text[]) THEN
        RAISE EXCEPTION 'Permission denied: Only email_alerts_enabled may be updated.';
    END IF;

    NEW.updated_at = now();
    RETURN NEW;
END;
$$;

REVOKE EXECUTE ON FUNCTION public.enforce_user_alert_settings_column_lock() FROM PUBLIC, anon;

DROP TRIGGER IF EXISTS trg_enforce_user_alert_settings_lock ON public.user_alert_settings;
CREATE TRIGGER trg_enforce_user_alert_settings_lock
    BEFORE UPDATE ON public.user_alert_settings
    FOR EACH ROW
    EXECUTE FUNCTION public.enforce_user_alert_settings_column_lock();

-- -----------------------------------------------------------------------------
-- 3. Additive Columns & Indexes on public.fall_events
-- -----------------------------------------------------------------------------
ALTER TABLE public.fall_events
    ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS email_sent_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS email_status TEXT CHECK (email_status IN ('PENDING', 'CLAIMED', 'SENT', 'FAILED', 'SKIPPED')),
    ADD COLUMN IF NOT EXISTS email_recipient TEXT,
    ADD COLUMN IF NOT EXISTS email_claimed_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS email_attempts INT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS email_error TEXT;

CREATE INDEX IF NOT EXISTS idx_fall_events_user_id ON public.fall_events (user_id);
CREATE INDEX IF NOT EXISTS idx_fall_events_user_created ON public.fall_events (user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_fall_events_email_claim ON public.fall_events (email_status, email_claimed_at, email_attempts);

-- -----------------------------------------------------------------------------
-- 4. INSERT Hardening & Rate Limiting on public.fall_events
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION public.enforce_fall_events_insert_rules()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = public, pg_temp
AS $$
DECLARE
    -- Rate limit thresholds (easily tunable)
    v_rate_limit_minute CONSTANT INT := 6;
    v_rate_limit_hour   CONSTANT INT := 40;
    v_recent_minute     INT;
    v_recent_hour       INT;
BEGIN
    -- Direct SQL maintenance users (postgres, supabase_admin in SQL Editor) and service_role bypass insert restrictions
    IF (auth.role() = 'service_role' OR (auth.role() IS NULL AND session_user IN ('postgres', 'supabase_admin'))) THEN
        RETURN NEW;
    END IF;

    -- Authenticated client must have valid session matching NEW.user_id
    IF (auth.uid() IS NULL OR NEW.user_id IS DISTINCT FROM auth.uid()) THEN
        RAISE EXCEPTION 'Permission denied: user_id must match authenticated session.';
    END IF;

    -- Per-user advisory transaction lock: serializes concurrent inserts for the same user,
    -- eliminating the race window between SELECT COUNT(*) and INSERT without locking out other users.
    PERFORM pg_advisory_xact_lock(hashtext('fall_insert_' || auth.uid()::text));

    -- Force server-controlled primary key and creation timestamp (clients cannot forge these)
    NEW.id := gen_random_uuid();
    NEW.created_at := now();

    -- Validate input constraints matching application specifications
    -- 1. Severity enum
    IF NEW.severity IS NULL OR NEW.severity NOT IN ('MILD', 'MODERATE', 'SEVERE') THEN
        RAISE EXCEPTION 'Invalid severity: must be MILD, MODERATE, or SEVERE.';
    END IF;

    -- 2. Fall score (Pending user live production query paste before setting strict range)
    IF NEW.fall_score IS NULL THEN
        RAISE EXCEPTION 'Invalid fall_score: cannot be null.';
    END IF;

    -- 3. Duration down (non-negative numeric seconds)
    IF NEW.duration_down IS NULL OR NEW.duration_down < 0 THEN
        RAISE EXCEPTION 'Invalid duration_down: must be non-negative.';
    END IF;

    -- 4. Person ID (small non-negative integer)
    IF NEW.person_id IS NULL OR NEW.person_id < 0 THEN
        RAISE EXCEPTION 'Invalid person_id: must be a non-negative integer.';
    END IF;

    -- 5. Notes length (max 500 characters)
    IF NEW.notes IS NOT NULL AND length(NEW.notes) > 500 THEN
        RAISE EXCEPTION 'Invalid notes: maximum length is 500 characters.';
    END IF;

    -- 6. Snapshot URL: NULL or strict match to project bucket UUID pattern
    IF NEW.snapshot_url IS NOT NULL AND NEW.snapshot_url !~ '^https://qbzuwqanxnkpeuioiiao\.supabase\.co/storage/v1/object/public/fall-snapshots/[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\.jpg$' THEN
        RAISE EXCEPTION 'Invalid snapshot_url: must match public fall-snapshots UUID URL pattern.';
    END IF;

    -- Force safe server-controlled defaults (clients cannot forge email or resolve states on insert)
    NEW.email_status := NULL;
    NEW.email_sent_at := NULL;
    NEW.email_recipient := NULL;
    NEW.email_claimed_at := NULL;
    NEW.email_attempts := 0;
    NEW.email_error := NULL;
    NEW.resolved := false;
    NEW.resolved_at := NULL;

    -- Rate limit check 1: Maximum 6 inserts per 60 seconds per user
    SELECT COUNT(*) INTO v_recent_minute
    FROM public.fall_events
    WHERE user_id = auth.uid()
      AND created_at > (now() - interval '60 seconds');

    IF (v_recent_minute >= v_rate_limit_minute) THEN
        RAISE EXCEPTION 'Rate limit exceeded: At most % fall events per 60 seconds.', v_rate_limit_minute;
    END IF;

    -- Rate limit check 2: Maximum 40 inserts per hour per user
    SELECT COUNT(*) INTO v_recent_hour
    FROM public.fall_events
    WHERE user_id = auth.uid()
      AND created_at > (now() - interval '1 hour');

    IF (v_recent_hour >= v_rate_limit_hour) THEN
        RAISE EXCEPTION 'Rate limit exceeded: At most % fall events per hour.', v_rate_limit_hour;
    END IF;

    RETURN NEW;
END;
$$;

REVOKE EXECUTE ON FUNCTION public.enforce_fall_events_insert_rules() FROM PUBLIC, anon;

DROP TRIGGER IF EXISTS trg_enforce_fall_events_insert ON public.fall_events;
CREATE TRIGGER trg_enforce_fall_events_insert
    BEFORE INSERT ON public.fall_events
    FOR EACH ROW
    EXECUTE FUNCTION public.enforce_fall_events_insert_rules();

-- -----------------------------------------------------------------------------
-- 5. UPDATE Protection on public.fall_events (Allowlist via JSONB-strip)
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION public.enforce_fall_events_user_column_lock()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = public, pg_temp
AS $$
BEGIN
    -- Direct SQL maintenance users (postgres, supabase_admin in SQL Editor) and service_role bypass column locks
    IF (auth.role() = 'service_role' OR (auth.role() IS NULL AND session_user IN ('postgres', 'supabase_admin'))) THEN
        RETURN NEW;
    END IF;

    -- For normal authenticated callers (both users and admins in web UI), only resolved, resolved_at,
    -- and notes may be modified. All other columns (id, person_id, user_id, severity, fall_score,
    -- duration_down, snapshot_url, created_at, email_*) are strictly immutable.
    IF (to_jsonb(NEW) - '{resolved, resolved_at, notes}'::text[] IS DISTINCT FROM
        to_jsonb(OLD) - '{resolved, resolved_at, notes}'::text[]) THEN
        RAISE EXCEPTION 'Permission denied: Only resolved, resolved_at, and notes may be updated.';
    END IF;

    RETURN NEW;
END;
$$;

REVOKE EXECUTE ON FUNCTION public.enforce_fall_events_user_column_lock() FROM PUBLIC, anon;

DROP TRIGGER IF EXISTS trg_enforce_fall_events_user_column_lock ON public.fall_events;
CREATE TRIGGER trg_enforce_fall_events_user_column_lock
    BEFORE UPDATE ON public.fall_events
    FOR EACH ROW
    EXECUTE FUNCTION public.enforce_fall_events_user_column_lock();

-- -----------------------------------------------------------------------------
-- 6. Replace Open RLS Policies on public.fall_events
-- -----------------------------------------------------------------------------
DROP POLICY IF EXISTS "Allow public read access" ON public.fall_events;
DROP POLICY IF EXISTS "Allow insert from api" ON public.fall_events;
DROP POLICY IF EXISTS "Allow update from api" ON public.fall_events;
DROP POLICY IF EXISTS "Admins see all; users see own fall events" ON public.fall_events;
DROP POLICY IF EXISTS "Authenticated users insert own events; admins and service role insert any" ON public.fall_events;
DROP POLICY IF EXISTS "Admins update all; users update own fall events" ON public.fall_events;

-- SELECT: Admins & service_role see all rows (including user_id IS NULL legacy rows);
-- normal users see only rows matching their own auth.uid().
CREATE POLICY "Admins see all; users see own fall events"
    ON public.fall_events
    FOR SELECT
    TO authenticated, service_role
    USING (
        auth.role() = 'service_role' OR public.is_admin() OR (auth.uid() = user_id)
    );

-- INSERT:
-- Service role inserts any event (including user_id IS NULL for Python pipeline).
-- Authenticated users insert events matching auth.uid() (further guarded by BEFORE INSERT trigger).
CREATE POLICY "Authenticated users insert own events; admins and service role insert any"
    ON public.fall_events
    FOR INSERT
    TO authenticated, service_role
    WITH CHECK (
        auth.role() = 'service_role' OR
        public.is_admin() OR
        (auth.uid() IS NOT NULL AND auth.uid() = user_id)
    );

-- UPDATE:
-- Service role and Admins can update any incident.
-- Users can update only their own incident (further guarded by BEFORE UPDATE trigger).
CREATE POLICY "Admins update all; users update own fall events"
    ON public.fall_events
    FOR UPDATE
    TO authenticated, service_role
    USING (
        auth.role() = 'service_role' OR
        public.is_admin() OR
        (auth.uid() IS NOT NULL AND auth.uid() = user_id)
    )
    WITH CHECK (
        auth.role() = 'service_role' OR
        public.is_admin() OR
        (auth.uid() IS NOT NULL AND auth.uid() = user_id)
    );

-- -----------------------------------------------------------------------------
-- 7. Safe, Non-Blocking Auto-Seeding Trigger on auth.users
-- -----------------------------------------------------------------------------
-- Rules:
-- 1. Never reads raw_user_meta_data for security decisions (untrusted user input).
-- 2. Grants 'admin' ONLY if:
--    - lower(trim(email)) = 'rohithkarthik25@gmail.com'
--    - email_confirmed_at IS NOT NULL
--    - Created via Google provider (raw_app_meta_data->>'provider' = 'google' or auth.identities)
-- 3. Grants 'user' to everyone else.
-- 4. Never blocks sign-in on exception.
CREATE OR REPLACE FUNCTION public.handle_new_auth_user()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
DECLARE
    v_clean_email TEXT;
    v_is_verified BOOLEAN := false;
    v_is_google BOOLEAN := false;
    v_assigned_role TEXT := 'user';
BEGIN
    v_clean_email := lower(trim(COALESCE(NEW.email, '')));

    -- Verification check: Only server-controlled email_confirmed_at
    IF (NEW.email_confirmed_at IS NOT NULL) THEN
        v_is_verified := true;
    END IF;

    -- Provider check: Only server-controlled app_metadata or auth.identities
    IF (NEW.raw_app_meta_data->>'provider' = 'google') THEN
        v_is_google := true;
    ELSIF EXISTS (SELECT 1 FROM auth.identities WHERE user_id = NEW.id AND provider = 'google') THEN
        v_is_google := true;
    END IF;

    -- Admin assignment rule: strictly verified Google account matching designated email
    IF (v_clean_email = 'rohithkarthik25@gmail.com' AND v_is_verified AND v_is_google) THEN
        v_assigned_role := 'admin';
    ELSE
        v_assigned_role := 'user';
    END IF;

    -- Safe idempotent role insertion (never demotes or overwrites an existing role)
    INSERT INTO public.user_roles (user_id, role, created_at)
    VALUES (NEW.id, v_assigned_role, now())
    ON CONFLICT (user_id) DO UPDATE
    SET role = CASE 
        WHEN v_clean_email = 'rohithkarthik25@gmail.com' AND v_is_verified AND v_is_google THEN 'admin'
        ELSE public.user_roles.role 
    END;

    -- Safe idempotent alert settings insertion
    IF v_clean_email <> '' THEN
        INSERT INTO public.user_alert_settings (user_id, email, email_alerts_enabled, created_at, updated_at)
        VALUES (NEW.id, v_clean_email, true, now(), now())
        ON CONFLICT (user_id) DO NOTHING;
    END IF;

    RETURN NEW;
EXCEPTION
    WHEN OTHERS THEN
        RAISE WARNING 'handle_new_auth_user notice for user % (%): %', NEW.id, NEW.email, SQLERRM;
        RETURN NEW;
END;
$$;

REVOKE EXECUTE ON FUNCTION public.handle_new_auth_user() FROM PUBLIC, anon;

DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
    AFTER INSERT OR UPDATE OF email_confirmed_at ON auth.users
    FOR EACH ROW
    EXECUTE FUNCTION public.handle_new_auth_user();

-- -----------------------------------------------------------------------------
-- 8. Backfill Existing Users in auth.users
-- -----------------------------------------------------------------------------
DO $$
DECLARE
    r RECORD;
    v_clean_email TEXT;
    v_is_verified BOOLEAN;
    v_is_google BOOLEAN;
    v_role TEXT;
BEGIN
    FOR r IN SELECT id, email, email_confirmed_at, raw_app_meta_data FROM auth.users LOOP
        v_clean_email := lower(trim(COALESCE(r.email, '')));
        v_is_verified := (r.email_confirmed_at IS NOT NULL);
        v_is_google := (
            (r.raw_app_meta_data->>'provider' = 'google')
            OR EXISTS (SELECT 1 FROM auth.identities WHERE user_id = r.id AND provider = 'google')
        );

        IF (v_clean_email = 'rohithkarthik25@gmail.com' AND v_is_verified AND v_is_google) THEN
            v_role := 'admin';
        ELSE
            v_role := 'user';
        END IF;

        -- Safe idempotent insert: never demotes or overwrites an existing role
        INSERT INTO public.user_roles (user_id, role, created_at)
        VALUES (r.id, v_role, now())
        ON CONFLICT (user_id) DO NOTHING;

        IF v_clean_email <> '' THEN
            INSERT INTO public.user_alert_settings (user_id, email, email_alerts_enabled, created_at, updated_at)
            VALUES (r.id, v_clean_email, true, now(), now())
            ON CONFLICT (user_id) DO NOTHING;
        END IF;
    END LOOP;
END $$;

-- -----------------------------------------------------------------------------
-- 9. Storage Hardening for fall-snapshots Bucket
-- -----------------------------------------------------------------------------
-- 1. Configure bucket parameters: 2 MB maximum file size and image/jpeg MIME type only
UPDATE storage.buckets
SET file_size_limit = 2097152, -- 2 MB
    allowed_mime_types = ARRAY['image/jpeg']
WHERE id = 'fall-snapshots';

-- 2. Tighten storage.objects policies:
-- - Upload: Authenticated users & service_role only, requiring exact UUID .jpg format.
-- - Public Select: REMOVED from storage.objects table to prevent bucket enumeration (.list()).
--   Direct URL fetches (public CDN) continue to work because bucket is public = true.
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_tables WHERE schemaname = 'storage' AND tablename = 'objects') THEN
        DROP POLICY IF EXISTS "Allow authenticated and service_role upload" ON storage.objects;
        DROP POLICY IF EXISTS "Allow public read snapshots" ON storage.objects;
        DROP POLICY IF EXISTS "Disallow anonymous delete snapshots" ON storage.objects;

        CREATE POLICY "Allow authenticated and service_role upload"
            ON storage.objects
            FOR INSERT
            TO authenticated, service_role
            WITH CHECK (
                bucket_id = 'fall-snapshots'
                AND name ~ '^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\.jpg$'
            );
    END IF;
END $$;
