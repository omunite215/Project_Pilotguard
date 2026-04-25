import { motion } from "motion/react";
import type { FrameResult } from "@/types/monitoring";

interface StatsPanelProps {
  frame: FrameResult | null;
}

export function StatsPanel({ frame }: StatsPanelProps) {
  const stats = [
    {
      label: "PERCLOS",
      value: frame ? `${frame.perclos_60s.toFixed(1)}%` : "—",
      warn: (frame?.perclos_60s ?? 0) > 30,
      icon: "👁",
    },
    {
      label: "Blinks/min",
      value: frame ? `${frame.blink_rate_pm.toFixed(0)}` : "—",
      warn: (frame?.blink_rate_pm ?? 17) < 8 || (frame?.blink_rate_pm ?? 17) > 30,
      icon: "💫",
    },
    {
      label: "MAR",
      value: frame?.mar != null ? frame.mar.toFixed(3) : "—",
      warn: (frame?.mar ?? 0) > 0.4,
      icon: "👄",
    },
    {
      label: "Confidence",
      value: frame ? `${(frame.confidence * 100).toFixed(0)}%` : "—",
      warn: frame ? frame.confidence < 0.5 : false,
      icon: "🎯",
    },
  ];

  return (
    <div className="grid grid-cols-2 gap-2">
      {stats.map((s, i) => (
        <motion.div
          key={s.label}
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 + i * 0.05 }}
          className={`group relative overflow-hidden rounded-xl border p-3 text-center transition-all hover:scale-[1.02] ${
            s.warn
              ? "border-alert-amber/40 bg-alert-amber/5"
              : "border-border bg-surface-raised"
          }`}
        >
          <p className="text-[10px] font-bold uppercase tracking-widest text-text-muted">
            {s.label}
          </p>
          <motion.p
            key={s.value}
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            className={`mt-1 font-mono text-lg font-black ${
              s.warn ? "text-alert-amber" : "text-text"
            }`}
          >
            {s.value}
          </motion.p>
        </motion.div>
      ))}
    </div>
  );
}
