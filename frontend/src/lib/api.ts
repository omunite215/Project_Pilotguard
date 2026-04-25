import type { Session, SessionDetail } from "@/types/session";
import type { Alert } from "@/types/monitoring";
import { API_BASE_URL } from "./constants";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.status} ${response.statusText}`);
  }

  return response.json() as Promise<T>;
}

export const api = {
  health: () =>
    request<{
      status: string;
      version: string;
      models_loaded: Record<string, boolean>;
      active_sessions: number;
    }>("/health"),

  startSession: (calibrationDuration = 30) =>
    request<{ session_id: string; started_at: string; status: string }>(
      "/session/start",
      {
        method: "POST",
        body: JSON.stringify({ calibration_duration: calibrationDuration }),
      },
    ),

  stopSession: (sessionId: string) =>
    request<SessionDetail>(`/session/${sessionId}/stop`, { method: "POST" }),

  getSession: (id: string) => request<SessionDetail>(`/session/${id}`),

  getSessions: (page = 1, pageSize = 20) =>
    request<{
      sessions: Session[];
      total: number;
      page: number;
      page_size: number;
    }>(`/sessions?page=${page}&page_size=${pageSize}`),

  getAlerts: (sessionId: string) => request<Alert[]>(`/alerts/${sessionId}`),
};
