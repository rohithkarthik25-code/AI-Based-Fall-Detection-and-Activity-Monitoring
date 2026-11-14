-- Supabase Schema for AI-Based Fall Detection and Activity Monitoring
-- Extensions
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_net";

-- 1. Table for fall incident events
CREATE TABLE IF NOT EXISTS public.fall_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    person_id INT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    severity TEXT NOT NULL CHECK (severity IN ('MILD', 'MODERATE', 'SEVERE')),
    fall_score NUMERIC(4,2) NOT NULL CHECK (fall_score >= 0 AND fall_score <= 1),
    duration_down NUMERIC(6,2) DEFAULT 0.0,
    snapshot_url TEXT,
    resolved BOOLEAN NOT NULL DEFAULT FALSE,
    resolved_at TIMESTAMPTZ,
    notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_fall_events_created_at ON public.fall_events (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_fall_events_unresolved ON public.fall_events (created_at DESC) WHERE resolved = FALSE;

-- 2. Table for daily activity metrics & ADL tracking
CREATE TABLE IF NOT EXISTS public.activity_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    person_id INT NOT NULL,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    standing_seconds INT NOT NULL DEFAULT 0,
    sitting_seconds INT NOT NULL DEFAULT 0,
    walking_seconds INT NOT NULL DEFAULT 0,
    lying_seconds INT NOT NULL DEFAULT 0,
    bending_seconds INT NOT NULL DEFAULT 0,
    sedentary_flag BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_activity_metrics_person_time ON public.activity_metrics (person_id, recorded_at DESC);

-- 3. Table for edge camera telemetry / heartbeat
CREATE TABLE IF NOT EXISTS public.camera_telemetry (
    camera_id TEXT PRIMARY KEY,
    last_heartbeat TIMESTAMPTZ NOT NULL DEFAULT now(),
    fps NUMERIC(5,2) NOT NULL DEFAULT 0.0,
    active_persons INT NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'ONLINE'
);

-- 4. Enable Row Level Security (RLS)
ALTER TABLE public.fall_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.activity_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.camera_telemetry ENABLE ROW LEVEL SECURITY;

-- 5. RLS Policies allowing anon reads & writes for the application
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'fall_events' AND policyname = 'Allow public read access') THEN
        CREATE POLICY "Allow public read access" ON public.fall_events FOR SELECT USING (true);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'fall_events' AND policyname = 'Allow insert from api') THEN
        CREATE POLICY "Allow insert from api" ON public.fall_events FOR INSERT WITH CHECK (true);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'fall_events' AND policyname = 'Allow update from api') THEN
        CREATE POLICY "Allow update from api" ON public.fall_events FOR UPDATE USING (true);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'activity_metrics' AND policyname = 'Allow public read access') THEN
        CREATE POLICY "Allow public read access" ON public.activity_metrics FOR SELECT USING (true);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'activity_metrics' AND policyname = 'Allow insert from api') THEN
        CREATE POLICY "Allow insert from api" ON public.activity_metrics FOR INSERT WITH CHECK (true);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'camera_telemetry' AND policyname = 'Allow public read access') THEN
        CREATE POLICY "Allow public read access" ON public.camera_telemetry FOR SELECT USING (true);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'camera_telemetry' AND policyname = 'Allow insert and update from api') THEN
        CREATE POLICY "Allow insert and update from api" ON public.camera_telemetry FOR ALL USING (true);
    END IF;
END $$;

-- 6. Enable Realtime on fall_events and camera_telemetry
ALTER PUBLICATION supabase_realtime ADD TABLE public.fall_events;
ALTER PUBLICATION supabase_realtime ADD TABLE public.camera_telemetry;

-- 7. Trigger to call Edge Function notify-fall
CREATE OR REPLACE FUNCTION public.trigger_fall_notification()
RETURNS TRIGGER AS $$
BEGIN
    PERFORM net.http_post(
        url := 'https://qbzuwqanxnkpeuioiiao.supabase.co/functions/v1/notify-fall',
        headers := '{"Content-Type": "application/json"}'::jsonb,
        body := json_build_object(
            'record', json_build_object(
                'id', NEW.id,
                'person_id', NEW.person_id,
                'created_at', NEW.created_at,
                'severity', NEW.severity,
                'fall_score', NEW.fall_score,
                'duration_down', NEW.duration_down,
                'snapshot_url', NEW.snapshot_url
            )
        )::jsonb
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE OR REPLACE TRIGGER on_fall_event_inserted
    AFTER INSERT ON public.fall_events
    FOR EACH ROW
    EXECUTE FUNCTION public.trigger_fall_notification();
