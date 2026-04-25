import { motion } from "motion/react";

interface EARIndicatorProps {
  earSmoothed: number | null;
  threshold: number;
  isCalibrating: boolean;
}

export function EARIndicator({ earSmoothed, threshold, isCalibrating }: EARIndicatorProps) {
  const ear = earSmoothed ?? 0;
  const pct = Math.min(100, Math.max(0, (ear / 0.5) * 100));
  const thresholdPct = Math.min(100, (threshold / 0.5) * 100);
  const isClosed = ear > 0 && ear < threshold;

  const statusColor = isCalibrating
    ? "text-primary"
    : isClosed ? "text-alert-red" : "text-alert-green";

  const statusBg = isCalibrating
    ? "bg-primary/10"
    : isClosed ? "bg-alert-red/10" : "bg-alert-green/10";

  const barColor = isClosed ? "bg-alert-red" : "bg-alert-green";

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: 0.2 }}
      className="rounded-2xl border border-border bg-surface-raised p-4 transition-colors"
    >
      <div className="mb-2 flex items-center justify-between">
        <span className="text-xs font-semibold text-text-secondary">Eye Aspect Ratio</span>
        <div className="flex items-center gap-2">
          <motion.span
            key={isCalibrating ? "cal" : isClosed ? "closed" : "open"}
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            className={`rounded-full px-2.5 py-0.5 text-[10px] font-bold ${statusBg} ${statusColor}`}
          >
            {isCalibrating ? "CALIBRATING" : isClosed ? "EYES CLOSED" : "EYES OPEN"}
          </motion.span>
          <span className={`font-mono text-sm font-bold ${statusColor}`}>
            {ear > 0 ? ear.toFixed(3) : "—"}
          </span>
        </div>
      </div>

      <div className="relative h-3 overflow-hidden rounded-full bg-surface-overlay">
        <motion.div
          className={`h-full rounded-full ${barColor}`}
          animate={{ width: `${pct}%` }}
          transition={{ type: "spring", stiffness: 200, damping: 25 }}
        />
        {!isCalibrating && threshold > 0 && (
          <div
            className="absolute top-0 h-full w-0.5 bg-text-muted"
            style={{ left: `${thresholdPct}%` }}
          />
        )}
      </div>
    </motion.div>
  );
}
