export const API_BASE_URL = "/api";

export function wsUrl(sessionId: string): string {
  const loc = window.location;
  const proto = loc.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${loc.host}/api/ws/${sessionId}`;
}

export const ALERT_THRESHOLDS = {
  advisory: 30,
  caution: 55,
  warning: 75,
} as const;

export const CAPTURE_FPS = 15;
export const JPEG_QUALITY = 0.7;
export const PERCLOS_WINDOW_SECONDS = 60;
export const RECONNECT_MAX_DELAY_MS = 30_000;
export const RECONNECT_BASE_DELAY_MS = 1_000;
export const CHART_THROTTLE_MS = 1_000;
