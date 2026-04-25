import { useEffect, useRef } from "react";
import { motion } from "motion/react";
import { useSettingsStore } from "@/stores/settingsStore";

interface TimePoint { t: number; ear: number }
interface FatiguePoint { t: number; score: number }

interface SessionTimelineProps {
  earHistory: TimePoint[];
  fatigueHistory: FatiguePoint[];
}

const H = 160;
const PAD = { top: 16, right: 8, bottom: 22, left: 36 };

export function SessionTimeline({ earHistory, fatigueHistory }: SessionTimelineProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef(0);
  const lastDraw = useRef(0);
  const dark = useSettingsStore((s) => s.darkMode);

  useEffect(() => {
    function draw() {
      const now = performance.now();
      if (now - lastDraw.current < 1000) { rafRef.current = requestAnimationFrame(draw); return; }
      lastDraw.current = now;

      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const dpr = window.devicePixelRatio || 1;
      const w = canvas.clientWidth;
      canvas.width = w * dpr;
      canvas.height = H * dpr;
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, w, H);

      const pw = w - PAD.left - PAD.right;
      const ph = H - PAD.top - PAD.bottom;

      const tNow = earHistory.length > 0 ? earHistory[earHistory.length - 1].t : 0;
      const tMin = tNow - 60;

      // Colors
      const gridColor = dark ? "#334155" : "#e2e8f0";
      const labelColor = dark ? "#64748b" : "#94a3b8";
      const bgColor = dark ? "#1e293b" : "#f8fafc";

      // Background
      ctx.fillStyle = bgColor;
      ctx.roundRect(PAD.left, PAD.top, pw, ph, 4);
      ctx.fill();

      // Threshold zones
      const zones = [
        { y: 30, color: dark ? "#422006" : "#fef9c3" },
        { y: 55, color: dark ? "#451a03" : "#fff7ed" },
        { y: 75, color: dark ? "#450a0a" : "#fef2f2" },
      ];
      for (const z of zones) {
        const zy = PAD.top + ph - (z.y / 100) * ph;
        ctx.fillStyle = z.color;
        ctx.fillRect(PAD.left, PAD.top, pw, zy - PAD.top);
      }

      // Grid
      ctx.strokeStyle = gridColor;
      ctx.lineWidth = 0.5;
      ctx.fillStyle = labelColor;
      ctx.font = "10px Inter, sans-serif";
      ctx.textAlign = "right";
      for (let v = 0; v <= 0.5; v += 0.1) {
        const y = PAD.top + ph - (v / 0.5) * ph;
        ctx.beginPath(); ctx.moveTo(PAD.left, y); ctx.lineTo(PAD.left + pw, y); ctx.stroke();
        ctx.fillText(v.toFixed(1), PAD.left - 4, y + 3);
      }

      // X labels
      ctx.textAlign = "center";
      for (let s = 0; s <= 60; s += 15) {
        ctx.fillText(`-${60 - s}s`, PAD.left + (s / 60) * pw, H - 4);
      }

      // EAR line (green)
      plotLine(ctx, earHistory.filter((d) => d.t >= tMin), tMin, tNow, pw, ph, 0, 0.5, "#22c55e", (d) => d.ear);

      // Fatigue line (red/orange)
      plotLine(ctx, fatigueHistory.filter((d) => d.t >= tMin), tMin, tNow, pw, ph, 0, 100, dark ? "#f87171" : "#ef4444", (d) => d.score);

      // Legend
      ctx.font = "bold 11px Inter, sans-serif";
      ctx.textAlign = "left";
      ctx.fillStyle = "#22c55e";
      ctx.fillText("● EAR", PAD.left + 6, PAD.top + 12);
      ctx.fillStyle = dark ? "#f87171" : "#ef4444";
      ctx.fillText("● Fatigue", PAD.left + 56, PAD.top + 12);

      rafRef.current = requestAnimationFrame(draw);
    }

    rafRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(rafRef.current);
  }, [earHistory, fatigueHistory, dark]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.25 }}
      className="rounded-2xl border border-border bg-surface-raised p-4 transition-colors"
    >
      <h3 className="mb-2 text-sm font-bold text-text">Session Timeline</h3>
      <canvas ref={canvasRef} style={{ width: "100%", height: H }} className="rounded-lg" />
    </motion.div>
  );
}

function plotLine<T>(
  ctx: CanvasRenderingContext2D,
  data: T[], tMin: number, tMax: number,
  pw: number, ph: number, vMin: number, vMax: number,
  color: string, getValue: (d: T) => number,
) {
  if (data.length < 2) return;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.lineJoin = "round";
  ctx.beginPath();
  for (let i = 0; i < data.length; i++) {
    const d = data[i] as T & { t: number };
    const x = PAD.left + ((d.t - tMin) / (tMax - tMin)) * pw;
    const y = PAD.top + ph - ((getValue(d) - vMin) / (vMax - vMin)) * ph;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
}
