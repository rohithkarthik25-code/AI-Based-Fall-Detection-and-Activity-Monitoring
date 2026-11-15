import "jsr:@supabase/functions-js/edge-runtime.d.ts";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

const TELEGRAM_BOT_TOKEN = Deno.env.get("TELEGRAM_BOT_TOKEN") || "9062818901:AAGIXsLlOJ-6V70_Wp_kTIAbIPSSD4-PFao";
const TELEGRAM_CHAT_ID = Deno.env.get("TELEGRAM_CHAT_ID") || "6666395760";

Deno.serve(async (req: Request) => {
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders });
  }

  try {
    const payload = await req.json();
    const event = payload.record || payload;

    // Do not dispatch Telegram alert if event is resolved, cancelled, or migrated historical data
    if (event.resolved === true || event.status === 'CANCELLED_BY_RESIDENT' || (event.notes && event.notes.toLowerCase().includes('migrated'))) {
      console.log("Skipping Telegram notification for resolved/migrated record:", event.id);
      return new Response(
        JSON.stringify({ success: true, skipped: true, reason: "Resolved or migrated historical record" }),
        { headers: { ...corsHeaders, "Content-Type": "application/json" }, status: 200 }
      );
    }

    const personId = event.person_id ?? "1";
    const severity = (event.severity ?? "SEVERE").toUpperCase();
    const score = event.fall_score !== undefined ? (Number(event.fall_score) * 100).toFixed(0) : "95";
    const duration = event.duration_down ? `${Number(event.duration_down).toFixed(1)}s` : "2.0s";
    const snapshotUrl = event.snapshot_url;
    const notes = event.notes || "Detected by AI Fall Detection System";
    const timestamp = event.created_at ? new Date(event.created_at).toLocaleString() : new Date().toLocaleString();

    const caption = `🚨 <b>EMERGENCY: FALL DETECTED!</b>\n` +
      `━━━━━━━━━━━━━━━━━━━\n` +
      `👤 <b>Person:</b> #${personId}\n` +
      `⚠️ <b>Severity:</b> ${severity}\n` +
      `📊 <b>Confidence Score:</b> ${score}%\n` +
      `⏱️ <b>Duration Down:</b> ${duration}\n` +
      `🕒 <b>Time:</b> ${timestamp}\n` +
      `📝 <b>Source:</b> ${notes}\n` +
      `━━━━━━━━━━━━━━━━━━━\n` +
      `<i>Action required immediately! Check dashboard now.</i>`;

    console.log("Dispatching Telegram notification to chat:", TELEGRAM_CHAT_ID);

    let telegramSent = false;
    let telegramError = null;

    if (snapshotUrl && snapshotUrl.startsWith("http")) {
      try {
        const photoResp = await fetch(`https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendPhoto`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            chat_id: TELEGRAM_CHAT_ID,
            photo: snapshotUrl,
            caption: caption,
            parse_mode: "HTML"
          })
        });
        const photoJson = await photoResp.json();
        if (photoJson.ok) {
          telegramSent = true;
          console.log("Telegram photo sent successfully:", photoJson.result?.message_id);
        } else {
          console.warn("Telegram sendPhoto failed, falling back to sendMessage:", photoJson.description);
        }
      } catch (err: any) {
        console.warn("Telegram photo error:", err.message);
      }
    }

    if (!telegramSent) {
      try {
        const msgResp = await fetch(`https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            chat_id: TELEGRAM_CHAT_ID,
            text: caption,
            parse_mode: "HTML"
          })
        });
        const msgJson = await msgResp.json();
        if (msgJson.ok) {
          telegramSent = true;
          console.log("Telegram message sent successfully:", msgJson.result?.message_id);
        } else {
          telegramError = msgJson.description;
          console.error("Telegram sendMessage error:", msgJson);
        }
      } catch (err: any) {
        telegramError = err.message;
        console.error("Telegram fetch error:", err);
      }
    }

    return new Response(
      JSON.stringify({
        success: true,
        telegram_sent: telegramSent,
        telegram_error: telegramError,
        severity: severity,
        person_id: personId,
        processed_at: new Date().toISOString()
      }),
      {
        headers: {
          ...corsHeaders,
          "Content-Type": "application/json"
        },
        status: 200
      }
    );
  } catch (err: any) {
    return new Response(
      JSON.stringify({ error: err.message }),
      {
        headers: {
          ...corsHeaders,
          "Content-Type": "application/json"
        },
        status: 400
      }
    );
  }
});
