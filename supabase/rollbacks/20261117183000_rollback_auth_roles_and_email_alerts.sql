-- =============================================================================
-- ROLLBACK SCRIPT: 20261117183000_rollback_auth_roles_and_email_alerts.sql
-- Location: supabase/rollbacks/ (Isolated from Supabase CLI migration directory)
-- =============================================================================
--
-- !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
-- WARNING / DATA LOSS NOTICE:
-- Executing this rollback script will PERMANENTLY:
--   1. Drop public.user_roles table (deleting all role assignments).
--   2. Drop public.user_alert_settings table (deleting alert preferences).
--   3. Drop user_id from public.fall_events (clearing all incident ownership).
--   4. Drop email_sent_at, email_status, email_recipient, email_claimed_at,
--      email_attempts, and email_error from public.fall_events.
--   5. Re-open public.fall_events to anonymous/public read, insert, and update!
--   6. Revert storage.objects policies to allow unauthenticated access and
--      remove the 2MB / image/jpeg bucket constraints.
--
-- ALWAYS perform a full database backup before running this script in production.
-- !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
--
-- =============================================================================
-- PRE-ROLLBACK INSPECTION / SNAPSHOT REFERENCE:
-- Run this read-only query in the Supabase SQL Editor BEFORE rolling back:
--
--   -- Inspect current storage bucket settings:
--   SELECT id, name, public, file_size_limit, allowed_mime_types
--   FROM storage.buckets
--   WHERE id = 'fall-snapshots';
--
--   -- Inspect current storage policies:
--   SELECT policyname, permissive, roles, cmd, qual, with_check
--   FROM pg_policies
--   WHERE schemaname = 'storage' AND tablename = 'objects'
--     AND (qual ILIKE '%fall-snapshots%' OR with_check ILIKE '%fall-snapshots%');
--
-- Original Storage Settings Prior to Migration:
--   Bucket: 'fall-snapshots', public = true, file_size_limit = NULL, allowed_mime_types = NULL
--   Original Policies:
--     "Allow public read snapshots": FOR SELECT TO public USING (bucket_id = 'fall-snapshots')
--     "Allow authenticated and service_role upload": FOR INSERT TO authenticated, service_role WITH CHECK (bucket_id = 'fall-snapshots')
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 1. Remove auth.users Trigger & Seeding Function
-- -----------------------------------------------------------------------------
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
DROP FUNCTION IF EXISTS public.handle_new_auth_user();

-- -----------------------------------------------------------------------------
-- 2. Remove Column Locks & Insert Hardening Triggers from public.fall_events
-- -----------------------------------------------------------------------------
DROP TRIGGER IF EXISTS trg_enforce_fall_events_user_column_lock ON public.fall_events;
DROP FUNCTION IF EXISTS public.enforce_fall_events_user_column_lock();

DROP TRIGGER IF EXISTS trg_enforce_fall_events_insert ON public.fall_events;
DROP FUNCTION IF EXISTS public.enforce_fall_events_insert_rules();

-- -----------------------------------------------------------------------------
-- 3. Remove Column Lock Trigger from public.user_alert_settings
-- -----------------------------------------------------------------------------
DROP TRIGGER IF EXISTS trg_enforce_user_alert_settings_lock ON public.user_alert_settings;
DROP FUNCTION IF EXISTS public.enforce_user_alert_settings_column_lock();

-- -----------------------------------------------------------------------------
-- 4. Revert Storage Policies & Bucket Constraints
-- -----------------------------------------------------------------------------
-- Revert bucket file size and MIME type limits to unrestricted defaults
UPDATE storage.buckets
SET file_size_limit = NULL,
    allowed_mime_types = NULL
WHERE id = 'fall-snapshots';

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_tables WHERE schemaname = 'storage' AND tablename = 'objects') THEN
        DROP POLICY IF EXISTS "Allow authenticated and service_role upload" ON storage.objects;
        DROP POLICY IF EXISTS "Allow public read snapshots" ON storage.objects;

        -- Restore original permissive upload policy (any filename in fall-snapshots)
        CREATE POLICY "Allow authenticated and service_role upload"
            ON storage.objects
            FOR INSERT
            TO authenticated, service_role
            WITH CHECK (bucket_id = 'fall-snapshots');

        -- Restore original public select policy
        CREATE POLICY "Allow public read snapshots"
            ON storage.objects
            FOR SELECT
            TO public
            USING (bucket_id = 'fall-snapshots');
    END IF;
END $$;

-- -----------------------------------------------------------------------------
-- 5. Revert fall_events RLS Policies to Original Open Public Access
-- -----------------------------------------------------------------------------
DROP POLICY IF EXISTS "Admins see all; users see own fall events" ON public.fall_events;
DROP POLICY IF EXISTS "Authenticated users insert own events; admins and service role insert any" ON public.fall_events;
DROP POLICY IF EXISTS "Admins update all; users update own fall events" ON public.fall_events;

-- Recreate the original open policies from 20261114045833_init_fall_detection.sql
CREATE POLICY "Allow public read access" ON public.fall_events FOR SELECT USING (true);
CREATE POLICY "Allow insert from api" ON public.fall_events FOR INSERT WITH CHECK (true);
CREATE POLICY "Allow update from api" ON public.fall_events FOR UPDATE USING (true);

-- -----------------------------------------------------------------------------
-- 6. Drop Added Columns & Indexes from public.fall_events
-- -----------------------------------------------------------------------------
DROP INDEX IF EXISTS public.idx_fall_events_email_claim;
DROP INDEX IF EXISTS public.idx_fall_events_user_created;
DROP INDEX IF EXISTS public.idx_fall_events_user_id;

ALTER TABLE public.fall_events
    DROP COLUMN IF EXISTS email_error,
    DROP COLUMN IF EXISTS email_attempts,
    DROP COLUMN IF EXISTS email_claimed_at,
    DROP COLUMN IF EXISTS email_recipient,
    DROP COLUMN IF EXISTS email_status,
    DROP COLUMN IF EXISTS email_sent_at,
    DROP COLUMN IF EXISTS user_id;

-- -----------------------------------------------------------------------------
-- 7. Drop user_alert_settings and user_roles Tables
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS public.user_alert_settings CASCADE;
DROP TABLE IF EXISTS public.user_roles CASCADE;

-- -----------------------------------------------------------------------------
-- 8. Drop Helper Functions
-- -----------------------------------------------------------------------------
DROP FUNCTION IF EXISTS public.is_admin();
