/**
 * AegisMotion AI / FallGuard AI - Central Configuration & Marketing Copy
 * All brand names, headlines, proof points, navigation links, and feature cards
 * are centralized here so they can be modified without altering layout markup.
 */

const APP_CONFIG = {
    brand: {
        name: "AegisMotion AI",             // Set to "FallGuard AI" or "AegisMotion AI" as needed
        shortName: "AegisMotion",
        tagline: "AI-Powered Assisted Living Safety",
        version: "2.0.0",
        logoGradient: "linear-gradient(135deg, #3F7FE0 0%, #7AA8F0 100%)",
        year: 2026,
    },
    hero: {
        eyebrow: "AI-powered assisted living safety",
        headingMain: "Detect falls in seconds.",
        headingAccent: "Protect who matters.",
        description: "Real-time pose estimation, a live 3D biomechanical skeleton and instant Telegram alerts, backed by a cloud incident ledger for every event.",
        ctaPrimary: {
            text: "Launch live monitor",
            href: "/#screen-live-monitor",
            title: "Launch real-time camera monitoring and pose estimation"
        },
        ctaSecondary: {
            text: "Upload a video",
            href: "/#screen-live-monitor?action=upload",
            title: "Upload a recorded video test file for fall detection"
        },
        notes: [
            {
                primary: "Live camera",
                secondary: "or recorded video"
            },
            {
                primary: "Instant alerts",
                secondary: "via Telegram"
            },
            {
                primary: "Cloud ledger",
                secondary: "with snapshots"
            }
        ]
    },
    nav: {
        links: [
            { label: "Live Monitor", href: "/#screen-live-monitor", targetScreen: "screen-live-monitor" },
            { label: "ADL Analytics", href: "/#screen-adl-analytics", targetScreen: "screen-adl-analytics" },
            { label: "Incident Ledger", href: "/#screen-incident-ledger", targetScreen: "screen-incident-ledger" },
            { label: "3D Viewer", href: "/#screen-skeleton-viewer", targetScreen: "screen-skeleton-viewer" }
        ],
        signInBtn: {
            text: "Sign in",
            action: "openSignInModal"
        },
        dashboardBtn: {
            text: "Open dashboard",
            href: "/"
        }
    },
    demoCards: {
        skeletonViewport: {
            title: "Interactive 3D Skeleton Viewport",
            badge: "Three.js (60 FPS)",
            balancePill: "Balance: STABLE | Margin: +8cm"
        },
        fallDetected: {
            status: "Fall detected",
            person: "Person #1 · SEVERE",
            confidenceLabel: "Confidence",
            confidenceValue: "92%"
        },
        mobility: {
            label: "MOBILITY",
            score: "96",
            status: "Optimal"
        },
        posture: {
            label: "CURRENT POSTURE",
            state: "Standing",
            angles: "Torso 0° · Hip / Knee 175° / 175°"
        },
        telegramAlert: {
            title: "Telegram alert sent",
            meta: "Caregiver notified · 2:04 AM"
        }
    },
    features: [
        {
            id: "pose",
            chipColor: "blue",
            title: "Pose estimation",
            desc: "Live posture and fall probability",
            href: "/#screen-live-monitor"
        },
        {
            id: "balance",
            chipColor: "purple",
            title: "3D balance analysis",
            desc: "Centre of mass and stability margin",
            href: "/#screen-skeleton-viewer"
        },
        {
            id: "adl",
            chipColor: "orange",
            title: "ADL analytics",
            desc: "Active vs sedentary time",
            href: "/#screen-adl-analytics"
        },
        {
            id: "ledger",
            chipColor: "red",
            title: "Incident ledger",
            desc: "Snapshots, severity, resolve",
            href: "/#screen-incident-ledger"
        }
    ]
};

// Export to window if in browser environment
if (typeof window !== "undefined") {
    window.APP_CONFIG = APP_CONFIG;
}

// Export for Node / CommonJS if needed
if (typeof module !== "undefined" && module.exports) {
    module.exports = APP_CONFIG;
}
