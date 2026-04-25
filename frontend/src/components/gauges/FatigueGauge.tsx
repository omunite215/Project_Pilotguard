import { motion, useSpring, useTransform } from "motion/react";

interface FatigueGaugeProps {
  score: number;
}

const RADIUS = 58;
const STROKE = 8;
const CIRC = 2 * Math.PI * RADIUS;
const SIZE = (RADIUS + STROKE + 4) * 2;

const ZONES = [
  { max: 30, color: "#22c55e", label: "NORMAL", glow: "0 0 24px #22c55e30" },
  { max: 55, color: "#eab308", label: "ADVISORY", glow: "0 0 24px #eab30830" },
  { max: 75, color: "#f59e0b", label: "CAUTION", glow: "0 0 24px #f59e0b30" },
  { max: 101, color: "#ef4444", label: "DANGER", glow: "0 0 28px #ef444450" },
];

function getZone(score: number) {
  return ZONES.find((z) => score < z.max) ?? ZONES[3];
}

export function FatigueGauge({ score }: FatigueGaugeProps) {
  const clamped = Math.max(0, Math.min(100, score));
  const zone = getZone(clamped);

  const springVal = useSpring(clamped, { stiffness: 80, damping: 20 });
  const displayScore = useTransform(springVal, (v) => Math.round(v));
  const dashOffset = useTransform(springVal, (v) => CIRC * (1 - v / 100));

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      className="flex flex-col items-center rounded-2xl border border-border bg-surface-raised p-5 transition-colors"
      style={{ boxShadow: zone.glow }}
    >
      <div className="relative" style={{ width: SIZE, height: SIZE }}>
        <svg width={SIZE} height={SIZE} className="-rotate-90">
          {/* Track */}
          <circle
            cx={SIZE / 2} cy={SIZE / 2} r={RADIUS}
            fill="none" stroke="var(--color-border)" strokeWidth={STROKE}
          />
          {/* Gradient def */}
          <defs>
            <linearGradient id="gauge-grad" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stopColor={zone.color} />
              <stop offset="100%" stopColor={zone.color} stopOpacity="0.5" />
            </linearGradient>
          </defs>
          {/* Active arc */}
          <motion.circle
            cx={SIZE / 2} cy={SIZE / 2} r={RADIUS}
            fill="none" stroke="url(#gauge-grad)" strokeWidth={STROKE}
            strokeLinecap="round" strokeDasharray={CIRC}
            style={{ strokeDashoffset: dashOffset }}
          />
        </svg>

        {/* Center content */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <motion.span
            className="text-4xl font-black tabular-nums"
            style={{ color: zone.color }}
          >
            {displayScore}
          </motion.span>
          <motion.span
            key={zone.label}
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-0.5 text-[10px] font-bold tracking-widest"
            style={{ color: zone.color }}
          >
            {zone.label}
          </motion.span>
        </div>
      </div>
      <span className="mt-2 text-xs font-semibold text-text-muted">Fatigue Score</span>
    </motion.div>
  );
}
