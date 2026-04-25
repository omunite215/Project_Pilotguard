import { createRoute } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { motion } from "motion/react";
import { rootRoute } from "./__root";
import { api } from "@/lib/api";
import type { Session } from "@/types/session";

function fatigueColor(score: number): string {
  if (score >= 75) return "text-alert-red";
  if (score >= 55) return "text-alert-amber";
  if (score >= 30) return "text-alert-yellow";
  return "text-alert-green";
}

function fatigueBg(score: number): string {
  if (score >= 75) return "bg-alert-red/10";
  if (score >= 55) return "bg-alert-amber/10";
  if (score >= 30) return "bg-alert-yellow/10";
  return "bg-alert-green/10";
}

function formatDuration(s: number): string {
  const m = Math.floor(s / 60);
  const sec = Math.round(s % 60);
  return m > 0 ? `${m}m ${sec}s` : `${sec}s`;
}

function HistoryPage() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["sessions"],
    queryFn: () => api.getSessions(),
    refetchInterval: 30_000,
  });

  const sessions: Session[] = data?.sessions ?? [];

  return (
    <div className="flex flex-col gap-6">
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h2 className="text-2xl font-black text-text">Session History</h2>
        <p className="mt-1 text-sm text-text-muted">
          Review past monitoring sessions and fatigue analytics
        </p>
      </motion.div>

      {isLoading && (
        <div className="flex flex-col gap-3">
          {[0, 1, 2].map((i) => (
            <div key={i} className="h-24 animate-pulse rounded-2xl bg-surface-overlay" />
          ))}
        </div>
      )}

      {error && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="rounded-2xl border border-alert-red/20 bg-alert-red/10 p-5 text-sm text-alert-red"
        >
          Failed to load sessions. Is the backend running?
        </motion.div>
      )}

      {!isLoading && sessions.length === 0 && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex flex-col items-center gap-3 py-16"
        >
          <span className="text-5xl">📊</span>
          <p className="text-text-muted">No sessions recorded yet</p>
          <p className="text-xs text-text-muted">Start monitoring to create your first session</p>
        </motion.div>
      )}

      {sessions.length > 0 && (
        <div className="flex flex-col gap-3">
          {sessions.map((session, i) => (
            <motion.div
              key={session.session_id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.05 }}
              whileHover={{ x: 4 }}
              className="flex flex-col gap-3 rounded-2xl border border-border bg-surface-raised p-5 transition-colors sm:flex-row sm:items-center sm:justify-between"
            >
              {/* Left: session info */}
              <div className="flex flex-col gap-1">
                <div className="flex items-center gap-2">
                  <span className="font-mono text-sm font-bold text-text">
                    {session.session_id}
                  </span>
                  <span
                    className={`rounded-full px-2.5 py-0.5 text-[10px] font-bold ${
                      session.status === "active"
                        ? "bg-alert-green/10 text-alert-green"
                        : "bg-surface-overlay text-text-muted"
                    }`}
                  >
                    {session.status}
                  </span>
                </div>
                <span className="text-xs text-text-muted">
                  {new Date(session.started_at).toLocaleString()}
                  {session.duration_seconds > 0 && ` — ${formatDuration(session.duration_seconds)}`}
                </span>
              </div>

              {/* Right: metrics */}
              <div className="flex items-center gap-5">
                <Metric label="frames" value={String(session.total_frames)} />
                <Metric label="avg fatigue" value={session.avg_fatigue_score.toFixed(1)} />
                <div className="text-center">
                  <p className={`rounded-lg px-2 py-0.5 font-mono text-sm font-black ${fatigueBg(session.max_fatigue_score)} ${fatigueColor(session.max_fatigue_score)}`}>
                    {session.max_fatigue_score.toFixed(1)}
                  </p>
                  <p className="mt-0.5 text-[10px] text-text-muted">max fatigue</p>
                </div>
                <Metric label="alerts" value={String(session.alert_count)} />
              </div>
            </motion.div>
          ))}
        </div>
      )}
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="text-center">
      <p className="font-mono text-sm font-bold text-text">{value}</p>
      <p className="text-[10px] text-text-muted">{label}</p>
    </div>
  );
}

export const historyRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/history",
  component: HistoryPage,
});
