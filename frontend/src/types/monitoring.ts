export interface FrameResult {
  frame_id: number;
  timestamp: number;
  face_detected: boolean;
  landmarks: [number, number][] | null;
  ear_left: number | null;
  ear_right: number | null;
  ear_avg: number | null;
  ear_smoothed: number | null;
  mar: number | null;
  state: "alert" | "drowsy" | "microsleep" | "unknown";
  fatigue_score: number;
  emotion: string | null;
  emotion_confidence: number | null;
  confidence: number;
  perclos_60s: number;
  blink_rate_pm: number;
  is_calibrating: boolean;
  calibration_progress: number;
  processing_time_ms: number;
  alert: Alert | null;
  is_locked: boolean;
  lock_remaining_seconds: number;
  lock_level: string;
  lock_progress: number;
  pilot_message: string | null;
  auto_stop: boolean;
  session_summary: string | null;
}

export interface Alert {
  level: "advisory" | "caution" | "warning";
  message: string;
  fatigue_score: number;
  timestamp: number;
}
