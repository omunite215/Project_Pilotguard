export interface Session {
  session_id: string;
  started_at: string;
  ended_at: string | null;
  duration_seconds: number;
  total_frames: number;
  avg_fatigue_score: number;
  max_fatigue_score: number;
  alert_count: number;
  status: "active" | "completed";
}

export interface SessionDetail extends Session {
  alerts?: SessionAlert[];
}

export interface SessionAlert {
  level: "advisory" | "caution" | "warning";
  message: string;
  fatigue_score: number;
  timestamp: number;
}
