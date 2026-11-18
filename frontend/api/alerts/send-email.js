// frontend/api/alerts/send-email.js
// Vercel Serverless Function for Emergency Fall Alert Email Dispatch
const nodemailer = require('nodemailer');

const PROJECT_REF = "qbzuwqanxnkpeuioiiao";
const SUPABASE_URL = process.env.SUPABASE_URL || `https://${PROJECT_REF}.supabase.co`;
const SUPABASE_ANON_KEY = process.env.SUPABASE_ANON_KEY || "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFienV3cWFueG5rcGV1aW9paWFvIiwicm9sZSI6ImFub24iLCJpYXQiOjE3OTQ2MjYxMzMsImV4cCI6MjExMDIwMjEzM30.cC9AZlgWMe3doYp2DtxjYRUGdDVn5L7YmTHpiKjRP0Q";
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;

const SNAPSHOT_REGEX = new RegExp(`^https://${PROJECT_REF}\\.supabase\\.co/storage/v1/object/public/fall-snapshots/([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\\.jpg)$`);

module.exports = async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method Not Allowed' });
  }

  const authHeader = req.headers.authorization;
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return res.status(401).json({ error: 'Missing or invalid Authorization header' });
  }
  const token = authHeader.replace('Bearer ', '').trim();

  const { event_id } = req.body || {};
  if (!event_id) {
    return res.status(400).json({ error: 'Missing required event_id' });
  }

  try {
    // 1. Validate JWT via Supabase Auth
    const userResp = await fetch(`${SUPABASE_URL}/auth/v1/user`, {
      headers: {
        'apikey': SUPABASE_ANON_KEY,
        'Authorization': `Bearer ${token}`
      }
    });
    if (!userResp.ok) {
      return res.status(401).json({ error: 'Unauthorized: Invalid Supabase session' });
    }
    const user = await userResp.json();
    const callerId = user.id;
    const callerEmail = user.email;

    if (!callerId || !callerEmail) {
      return res.status(401).json({ error: 'Unauthorized: User identity not found' });
    }

    if (!SUPABASE_SERVICE_ROLE_KEY) {
      return res.status(500).json({ error: 'Server misconfiguration: SUPABASE_SERVICE_ROLE_KEY is required' });
    }

    // 1b. Suspended accounts cannot trigger alerts (RLS is the primary enforcement).
    const blockResp = await fetch(`${SUPABASE_URL}/rest/v1/user_roles?user_id=eq.${encodeURIComponent(callerId)}&select=blocked_at`, {
      headers: {
        'apikey': SUPABASE_SERVICE_ROLE_KEY,
        'Authorization': `Bearer ${SUPABASE_SERVICE_ROLE_KEY}`
      }
    });
    if (blockResp.ok) {
      const rows = await blockResp.json();
      if (Array.isArray(rows) && rows.some(r => r && r.blocked_at)) {
        return res.status(403).json({ error: 'Forbidden: This account has been suspended by an administrator.' });
      }
    }

    // 2. Atomic Claim Query via Supabase PostgREST
    const twoMinutesAgo = new Date(Date.now() - 2 * 60 * 1000).toISOString();
    const claimUrl = `${SUPABASE_URL}/rest/v1/fall_events?id=eq.${event_id}&email_sent_at=is.null&or=(email_claimed_at.is.null,email_claimed_at.lt.${twoMinutesAgo})`;

    const claimResp = await fetch(claimUrl, {
      method: 'PATCH',
      headers: {
        'apikey': SUPABASE_SERVICE_ROLE_KEY,
        'Authorization': `Bearer ${SUPABASE_SERVICE_ROLE_KEY}`,
        'Content-Type': 'application/json',
        'Prefer': 'return=representation'
      },
      body: JSON.stringify({
        email_status: 'CLAIMED',
        email_claimed_at: new Date().toISOString()
      })
    });

    const claimedEvents = await claimResp.json();
    if (!claimedEvents || claimedEvents.length === 0) {
      return res.status(200).json({
        status: 'skipped',
        message: 'Event is already claimed, sent, or within 2-minute lock.'
      });
    }

    const event = claimedEvents[0];

    // 3. Verify Ownership or Admin
    let isAdmin = false;
    const roleResp = await fetch(`${SUPABASE_URL}/rest/v1/user_roles?user_id=eq.${callerId}&role=eq.admin&select=role`, {
      headers: {
        'apikey': SUPABASE_SERVICE_ROLE_KEY,
        'Authorization': `Bearer ${SUPABASE_SERVICE_ROLE_KEY}`
      }
    });
    if (roleResp.ok) {
      const roles = await roleResp.json();
      isAdmin = roles && roles.length > 0;
    }

    if (event.user_id !== callerId && !isAdmin) {
      // Revert claim
      await fetch(`${SUPABASE_URL}/rest/v1/fall_events?id=eq.${event_id}`, {
        method: 'PATCH',
        headers: {
          'apikey': SUPABASE_SERVICE_ROLE_KEY,
          'Authorization': `Bearer ${SUPABASE_SERVICE_ROLE_KEY}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          email_status: 'FAILED',
          email_error: 'Forbidden: caller does not own event'
        })
      });
      return res.status(403).json({ error: 'Forbidden: You do not have permission to trigger alerts for this incident' });
    }

    // 4. Validate snapshot URL & Download via Storage API (SSRF Safe)
    let snapshotBuffer = null;
    let snapshotFilename = null;
    if (event.snapshot_url) {
      const match = SNAPSHOT_REGEX.exec(event.snapshot_url.trim());
      if (match) {
        snapshotFilename = match[1];
        const storageResp = await fetch(`${SUPABASE_URL}/storage/v1/object/fall-snapshots/${snapshotFilename}`, {
          headers: {
            'apikey': SUPABASE_SERVICE_ROLE_KEY,
            'Authorization': `Bearer ${SUPABASE_SERVICE_ROLE_KEY}`
          }
        });
        if (storageResp.ok) {
          snapshotBuffer = Buffer.from(await storageResp.arrayBuffer());
        }
      }
    }

    // 5. Send Email via SMTP
    const transporter = nodemailer.createTransport({
      host: process.env.SMTP_HOST || 'smtp.gmail.com',
      port: parseInt(process.env.SMTP_PORT || '587', 10),
      secure: false, // TLS via STARTTLS
      auth: {
        user: process.env.SMTP_USER,
        pass: process.env.SMTP_APP_PASSWORD
      }
    });

    const severity = event.severity || 'SEVERE';
    const score = event.fall_score !== undefined ? `${(Number(event.fall_score) * 100).toFixed(0)}%` : '95%';
    const duration = event.duration_down ? `${Number(event.duration_down).toFixed(1)}s` : '5.0s';
    const personId = event.person_id ?? 1;
    const timeStr = event.created_at ? new Date(event.created_at).toLocaleString() : new Date().toLocaleString();
    const fromName = process.env.ALERT_FROM_NAME || 'FallGuard AI';

    const attachments = [];
    let cidImageHtml = '';
    if (snapshotBuffer) {
      attachments.push({
        filename: 'fall_snapshot.jpg',
        content: snapshotBuffer,
        cid: 'fall_snapshot_cid'
      });
      cidImageHtml = `<div style="text-align: center; margin: 20px 0;"><img src="cid:fall_snapshot_cid" alt="Fall Snapshot" style="max-width: 100%; border-radius: 8px; border: 2px solid #ef4444;" /></div>`;
    }

    const mailOptions = {
      from: `"${fromName}" <${process.env.SMTP_USER}>`,
      to: callerEmail,
      subject: `🚨 [EMERGENCY ALERT] Fall Detected: Person #${personId} (${severity})`,
      html: `
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; border: 1px solid #e2e8f0; border-radius: 8px; overflow: hidden;">
          <div style="background-color: #ef4444; color: white; padding: 18px 24px; text-align: center;">
            <h1 style="margin: 0; font-size: 20px; letter-spacing: 0.5px;">EMERGENCY FALL DETECTED</h1>
          </div>
          <div style="padding: 24px; background-color: #ffffff; color: #1e293b;">
            <p style="font-size: 16px; margin-top: 0;">An unrecovered fall was detected on your monitored device.</p>
            <table style="width: 100%; border-collapse: collapse; margin: 20px 0;">
              <tr><td style="padding: 8px 0; font-weight: bold;">Person:</td><td>#${personId}</td></tr>
              <tr><td style="padding: 8px 0; font-weight: bold;">Severity:</td><td style="color: #ef4444; font-weight: bold;">${severity}</td></tr>
              <tr><td style="padding: 8px 0; font-weight: bold;">Confidence:</td><td>${score}</td></tr>
              <tr><td style="padding: 8px 0; font-weight: bold;">Duration Down:</td><td>${duration}</td></tr>
              <tr><td style="padding: 8px 0; font-weight: bold;">Time:</td><td>${timeStr}</td></tr>
            </table>
            ${cidImageHtml}
            <div style="background-color: #f8fafc; border-left: 4px solid #3b82f6; padding: 12px; margin-top: 20px; font-size: 13px; color: #64748b;">
              This emergency alert was dispatched to your Google account <strong>${callerEmail}</strong>.
            </div>
          </div>
        </div>
      `,
      attachments: attachments
    };

    await transporter.sendMail(mailOptions);

    // 6. Mark SENT
    await fetch(`${SUPABASE_URL}/rest/v1/fall_events?id=eq.${event_id}`, {
      method: 'PATCH',
      headers: {
        'apikey': SUPABASE_SERVICE_ROLE_KEY,
        'Authorization': `Bearer ${SUPABASE_SERVICE_ROLE_KEY}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        email_status: 'SENT',
        email_sent_at: new Date().toISOString(),
        email_recipient: callerEmail,
        email_error: null
      })
    });

    return res.status(200).json({ status: 'sent', recipient: callerEmail, event_id });
  } catch (err) {
    console.error('Serverless send-email error:', err);
    try {
      await fetch(`${SUPABASE_URL}/rest/v1/fall_events?id=eq.${event_id}`, {
        method: 'PATCH',
        headers: {
          'apikey': SUPABASE_SERVICE_ROLE_KEY,
          'Authorization': `Bearer ${SUPABASE_SERVICE_ROLE_KEY}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          email_status: 'FAILED',
          email_error: String(err && err.message ? err.message : err)
        })
      });
    } catch (_) {}
    return res.status(500).json({ error: 'Failed to process email alert', details: String(err) });
  }
};
