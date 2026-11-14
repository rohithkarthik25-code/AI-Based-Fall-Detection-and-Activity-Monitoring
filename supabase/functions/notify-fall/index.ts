import "jsr:@supabase/functions-js/edge-runtime.d.ts";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

Deno.serve(async (req: Request) => {
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders });
  }

  try {
    const payload = await req.json();
    // Accommodate both Supabase DB webhook { record: { ... } } and direct POST { person_id: ... }
    const event = payload.record || payload;

    const personId = event.person_id ?? "Unknown";
    const severity = event.severity ?? "UNKNOWN";
    const score = event.fall_score ?? 0.0;
    const snapshotUrl = event.snapshot_url || "No snapshot available";
    const timestamp = event.created_at || new Date().toISOString();

    const alertMessage = `🚨 [FALL DETECTED] Person #${personId} | Severity: ${severity} | Score: ${score} | Snapshot: ${snapshotUrl} | Time: ${timestamp}`;
    console.log(alertMessage);

    return new Response(
      JSON.stringify({
        success: true,
        alert_sent: true,
        severity: severity,
        person_id: personId,
        message: alertMessage,
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
