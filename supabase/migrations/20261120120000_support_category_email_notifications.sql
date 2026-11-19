-- Migration: Alerts are email-only. Rename support ticket category
-- 'Telegram Notifications' -> 'Email Notifications'.

ALTER TABLE public.support_requests DROP CONSTRAINT IF EXISTS support_requests_category_check;

UPDATE public.support_requests
    SET category = 'Email Notifications'
    WHERE category = 'Telegram Notifications';

ALTER TABLE public.support_requests ADD CONSTRAINT support_requests_category_check CHECK (category IN (
    'Camera Issues',
    'Video Upload Issues',
    'Fall Detection Errors',
    'Multiple Person Detection',
    '3D Visualization Issues',
    'Biomechanical Measurement Issues',
    'Performance Issues',
    'Email Notifications',
    'Supabase / Database Issues',
    'Other'
));
