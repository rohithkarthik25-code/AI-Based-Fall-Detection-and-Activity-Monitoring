-- Migration: Create support_requests table for AegisMotion Help & Support portal
-- Preserves all existing tables (fall_events, activity_metrics, camera_telemetry)

CREATE TABLE IF NOT EXISTS public.support_requests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    full_name TEXT NOT NULL,
    email TEXT NOT NULL,
    category TEXT NOT NULL CHECK (category IN (
        'Camera Issues',
        'Video Upload Issues',
        'Fall Detection Errors',
        'Multiple Person Detection',
        '3D Visualization Issues',
        'Biomechanical Measurement Issues',
        'Performance Issues',
        'Telegram Notifications',
        'Supabase / Database Issues',
        'Other'
    )),
    subject TEXT NOT NULL,
    description TEXT NOT NULL,
    attachment_url TEXT,
    status TEXT NOT NULL DEFAULT 'OPEN' CHECK (status IN ('OPEN', 'IN_REVIEW', 'RESOLVED', 'CLOSED')),
    priority TEXT NOT NULL DEFAULT 'NORMAL' CHECK (priority IN ('LOW', 'NORMAL', 'HIGH', 'CRITICAL'))
);

CREATE INDEX IF NOT EXISTS idx_support_requests_created_at ON public.support_requests (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_support_requests_category ON public.support_requests (category);
CREATE INDEX IF NOT EXISTS idx_support_requests_status ON public.support_requests (status);

-- Enable Row Level Security (RLS)
ALTER TABLE public.support_requests ENABLE ROW LEVEL SECURITY;

-- RLS Policies: Allow public anonymous insertion for support requests
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'support_requests' AND policyname = 'Allow public insert support requests') THEN
        CREATE POLICY "Allow public insert support requests" ON public.support_requests FOR INSERT WITH CHECK (true);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'support_requests' AND policyname = 'Allow public read support requests') THEN
        CREATE POLICY "Allow public read support requests" ON public.support_requests FOR SELECT USING (true);
    END IF;
END $$;
