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
