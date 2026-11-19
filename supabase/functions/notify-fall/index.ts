import "jsr:@supabase/functions-js/edge-runtime.d.ts";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

// Fall alerts are delivered by email only. The email itself is sent by
// POST /api/alerts/send-email; this function just acknowledges new fall events.
Deno.serve(async (req: Request) => {
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders });
  }

  try {
    const payload = await req.json();
    const event = payload.record || payload;

    // Do not process alert if event is resolved, cancelled, or migrated historical data
    if (event.resolved === true || event.status === 'CANCELLED_BY_RESIDENT' || (event.notes && event.notes.toLowerCase().includes('migrated'))) {
      console.log("Skipping email alert for resolved/migrated record:", event.id);
      return new Response(
        JSON.stringify({ success: true, skipped: true, reason: "Resolved or migrated historical record" }),
        { headers: { ...corsHeaders, "Content-Type": "application/json" }, status: 200 }
      );
    }

    const personId = event.person_id ?? "1";
    const severity = (event.severity ?? "SEVERE").toUpperCase();

    console.log("Fall event received; email alert is generated via /api/alerts/send-email:", event.id);

    return new Response(
      JSON.stringify({
        success: true,
        alert_channel: "email",
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
