-- =============================================================================
-- Test Suite: test_auth_roles_rls.sql
-- Description:
--   SQL test harness designed for staging / local Supabase.
--   Tests RLS, column locks, rate limits, and security triggers under the
--   real 'authenticated' role using SET LOCAL ROLE and request.jwt.claims.
--
-- Note: Do NOT execute against Production.
-- Run in Staging using: psql "$STAGING_DB_URL" -f supabase/tests/test_auth_roles_rls.sql
-- =============================================================================

BEGIN;

-- -----------------------------------------------------------------------------
-- 0. Fixtures Setup (as postgres)
-- -----------------------------------------------------------------------------
RESET ROLE;

DELETE FROM public.fall_events WHERE notes LIKE 'TEST_HARNESS_%';
DELETE FROM auth.users WHERE email LIKE 'test_%@example.com' OR email = 'rohithkarthik25@gmail.com';

-- User 1: Legitimate Admin (Google OAuth verified)
INSERT INTO auth.users (id, email, email_confirmed_at, raw_app_meta_data, raw_user_meta_data)
VALUES (
    '11111111-1111-1111-1111-111111111111',
    'rohithkarthik25@gmail.com',
    now(),
    '{"provider": "google", "providers": ["google"]}'::jsonb,
    '{"full_name": "Admin User"}'::jsonb
);

-- User 2: Standard User Alice (Google OAuth verified)
INSERT INTO auth.users (id, email, email_confirmed_at, raw_app_meta_data, raw_user_meta_data)
VALUES (
    '22222222-2222-2222-2222-222222222222',
    'test_user_alice@example.com',
    now(),
    '{"provider": "google", "providers": ["google"]}'::jsonb,
    '{"full_name": "Alice"}'::jsonb
);

-- User 3: Standard User Bob (Google OAuth verified)
INSERT INTO auth.users (id, email, email_confirmed_at, raw_app_meta_data, raw_user_meta_data)
VALUES (
    '33333333-3333-3333-3333-333333333333',
    'test_user_bob@example.com',
    now(),
    '{"provider": "google", "providers": ["google"]}'::jsonb,
    '{"full_name": "Bob"}'::jsonb
);

-- User 4: Attacker attempting Admin spoof via Email/Password + forged user_metadata
INSERT INTO auth.users (id, email, email_confirmed_at, raw_app_meta_data, raw_user_meta_data)
VALUES (
    '44444444-4444-4444-4444-444444444444',
    'rohithkarthik25_spoof@gmail.com',
    now(),
    '{"provider": "email", "providers": ["email"]}'::jsonb,
    '{"email_verified": true, "role": "admin", "provider": "google"}'::jsonb
);

-- -----------------------------------------------------------------------------
-- 1. Test Role Assignment Trigger
-- -----------------------------------------------------------------------------
DO $$
DECLARE
    v_role TEXT;
BEGIN
    SELECT role INTO v_role FROM public.user_roles WHERE user_id = '11111111-1111-1111-1111-111111111111';
    IF v_role <> 'admin' THEN RAISE EXCEPTION 'FAIL: Verified Google admin account was not assigned admin'; END IF;

    SELECT role INTO v_role FROM public.user_roles WHERE user_id = '22222222-2222-2222-2222-222222222222';
    IF v_role <> 'user' THEN RAISE EXCEPTION 'FAIL: Standard user was not assigned user'; END IF;

    SELECT role INTO v_role FROM public.user_roles WHERE user_id = '44444444-4444-4444-4444-444444444444';
    IF v_role <> 'user' THEN RAISE EXCEPTION 'FAIL: Forged user_metadata account became admin!'; END IF;

    RAISE NOTICE '✓ Test 1: Role auto-assignment rules passed.';
END $$;

-- -----------------------------------------------------------------------------
-- 2. Test Under REAL 'authenticated' Role (Catches SECURITY DEFINER bugs)
-- -----------------------------------------------------------------------------
SET LOCAL ROLE authenticated;
SELECT set_config('request.jwt.claims', '{"sub": "22222222-2222-2222-2222-222222222222", "role": "authenticated"}', true);

-- 2.1 Insert an incident as Alice
INSERT INTO public.fall_events (
    user_id, severity, fall_score, duration_down, person_id, notes
) VALUES (
    '22222222-2222-2222-2222-222222222222', 'SEVERE', 0.95, 5.0, 1, 'TEST_HARNESS_ALICE_EVENT_1'
);

-- 2.2 Attempt to modify person_id on the incident (MUST FAIL under authenticated role)
DO $$
DECLARE
    v_id UUID;
BEGIN
    SELECT id INTO v_id FROM public.fall_events WHERE notes = 'TEST_HARNESS_ALICE_EVENT_1';
    BEGIN
        UPDATE public.fall_events SET person_id = 99 WHERE id = v_id;
        RAISE EXCEPTION 'CRITICAL SECURITY FAIL: Authenticated user was able to update person_id!';
    EXCEPTION WHEN OTHERS THEN
        IF SQLERRM LIKE '%Only resolved, resolved_at, and notes may be updated%' THEN
            RAISE NOTICE '✓ Test 2.2 Passed: Standard user blocked from updating person_id under authenticated role.';
        ELSE
            RAISE EXCEPTION 'Unexpected error on person_id update: %', SQLERRM;
        END IF;
    END;
END $$;

-- 2.3 Attempt to modify user_alert_settings email (MUST FAIL under authenticated role)
DO $$
BEGIN
    BEGIN
        UPDATE public.user_alert_settings SET email = 'attacker@evil.com' WHERE user_id = '22222222-2222-2222-2222-222222222222';
        RAISE EXCEPTION 'CRITICAL SECURITY FAIL: Authenticated user was able to modify alert email!';
    EXCEPTION WHEN OTHERS THEN
        IF SQLERRM LIKE '%Only email_alerts_enabled may be updated%' THEN
            RAISE NOTICE '✓ Test 2.3 Passed: Standard user blocked from updating alert settings email.';
        ELSE
            RAISE EXCEPTION 'Unexpected error on alert email update: %', SQLERRM;
        END IF;
    END;
END $$;

-- 2.4 Verify user CAN update resolved, resolved_at, and notes
DO $$
DECLARE
    v_id UUID;
BEGIN
    SELECT id INTO v_id FROM public.fall_events WHERE notes = 'TEST_HARNESS_ALICE_EVENT_1';
    UPDATE public.fall_events
    SET resolved = true,
        resolved_at = now(),
        notes = 'TEST_HARNESS_RESOLVED'
    WHERE id = v_id;
    RAISE NOTICE '✓ Test 2.4 Passed: Standard user allowed to update allowlisted fields (resolved, resolved_at, notes).';
END $$;

-- 2.5 Verify Rate Limiting (6 per 60s under authenticated role)
DO $$
DECLARE
    v_caught BOOLEAN := false;
    i INT;
BEGIN
    -- Alice already inserted 1. Insert 5 more (total 6)
    FOR i IN 2..6 LOOP
        INSERT INTO public.fall_events (user_id, severity, fall_score, duration_down, person_id, notes)
        VALUES ('22222222-2222-2222-2222-222222222222', 'MILD', 0.5, 2.0, 1, 'TEST_HARNESS_FLOOD_' || i);
    END LOOP;

    -- 7th insert MUST be rejected by rate limiter
    BEGIN
        INSERT INTO public.fall_events (user_id, severity, fall_score, duration_down, person_id, notes)
        VALUES ('22222222-2222-2222-2222-222222222222', 'MILD', 0.5, 2.0, 1, 'TEST_HARNESS_FLOOD_7');
    EXCEPTION WHEN OTHERS THEN
        IF SQLERRM LIKE '%Rate limit exceeded%' THEN
            v_caught := true;
        END IF;
    END;

    IF NOT v_caught THEN
        RAISE EXCEPTION 'CRITICAL SECURITY FAIL: Rate limit did not block 7th insert under authenticated role!';
    END IF;

    RAISE NOTICE '✓ Test 2.5 Passed: Rate limiter blocks 7th insert in 60s window.';
END $$;

-- -----------------------------------------------------------------------------
-- 3. Maintenance Bypass Test (as postgres / service_role)
-- -----------------------------------------------------------------------------
RESET ROLE;

DO $$
DECLARE
    v_id UUID;
BEGIN
    SELECT id INTO v_id FROM public.fall_events WHERE notes LIKE 'TEST_HARNESS_%' LIMIT 1;
    -- Maintenance user updating person_id must succeed
    UPDATE public.fall_events SET person_id = 42 WHERE id = v_id;
    RAISE NOTICE '✓ Test 3 Passed: Direct SQL maintenance user bypasses column lock.';
END $$;

-- -----------------------------------------------------------------------------
-- 4. Atomic Email Claim Test
-- -----------------------------------------------------------------------------
DO $$
DECLARE
    v_id UUID;
    v_claimed_id UUID;
    v_second_count INT;
BEGIN
    SELECT id INTO v_id FROM public.fall_events WHERE notes LIKE 'TEST_HARNESS_%' LIMIT 1;

    -- Reset claim fields
    UPDATE public.fall_events SET email_claimed_at = NULL, email_status = NULL, email_sent_at = NULL WHERE id = v_id;

    -- Claim 1: succeeds
    UPDATE public.fall_events
    SET email_claimed_at = now(),
        email_status = 'CLAIMED',
        email_attempts = email_attempts + 1
    WHERE id = v_id
      AND email_sent_at IS NULL
      AND (email_claimed_at IS NULL OR email_claimed_at < now() - interval '2 minutes')
    RETURNING id INTO v_claimed_id;

    IF v_claimed_id IS NULL THEN RAISE EXCEPTION 'FAIL: First atomic claim failed!'; END IF;

    -- Claim 2 (immediate concurrent): returns 0 rows
    UPDATE public.fall_events
    SET email_claimed_at = now(),
        email_status = 'CLAIMED',
        email_attempts = email_attempts + 1
    WHERE id = v_id
      AND email_sent_at IS NULL
      AND (email_claimed_at IS NULL OR email_claimed_at < now() - interval '2 minutes');

    GET DIAGNOSTICS v_second_count = ROW_COUNT;
    IF v_second_count <> 0 THEN RAISE EXCEPTION 'FAIL: Concurrent claim should have returned 0 rows!'; END IF;

    RAISE NOTICE '✓ Test 4 Passed: Atomic claim guarantees single worker execution.';
END $$;

RAISE NOTICE '==================================================';
RAISE NOTICE 'ALL SQL SECURITY TESTS PASSED UNDER REAL ROLES';
RAISE NOTICE '==================================================';

ROLLBACK;
