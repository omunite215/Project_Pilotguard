import { z } from "zod/v4";

export const alertSchema = z.object({
  level: z.enum(["advisory", "caution", "warning"]),
  message: z.string(),
  timestamp: z.number(),
});

export const frameResultSchema = z.object({
  frame_id: z.number(),
  timestamp: z.number(),
  landmarks: z.array(z.tuple([z.number(), z.number()])),
  ear_left: z.number(),
  ear_right: z.number(),
  ear_avg: z.number(),
  ear_smoothed: z.number(),
  mar: z.number(),
  state: z.enum(["alert", "drowsy", "microsleep"]),
  fatigue_score: z.number(),
  emotion: z.string(),
  confidence: z.number(),
  perclos_60s: z.number(),
  blink_rate_pm: z.number(),
  alert: alertSchema.nullable(),
});

export const sessionSchema = z.object({
  id: z.string(),
  start_time: z.string(),
  end_time: z.string().nullable(),
  avg_fatigue: z.number(),
  max_fatigue: z.number(),
  alert_count: z.number(),
});
