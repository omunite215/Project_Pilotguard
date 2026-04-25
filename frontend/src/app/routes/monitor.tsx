import { createRoute } from "@tanstack/react-router";
import { useCallback, useEffect, useRef, useState } from "react";
import { motion } from "motion/react";
import { rootRoute } from "./__root";
import { useMonitorStore } from "@/stores/monitorStore";
import { useAlertStore } from "@/stores/alertStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useWebSocket } from "@/features/monitoring/useWebSocket";
import { useFrameCapture } from "@/features/monitoring/useFrameCapture";
import { useAudioAlert } from "@/features/monitoring/useAudioAlert";
import { useMediaDevices } from "@/hooks/useMediaDevices";
import { api } from "@/lib/api";
import { VideoFeed } from "@/components/video/VideoFeed";
import { FatigueGauge } from "@/components/gauges/FatigueGauge";
import { EARIndicator } from "@/components/gauges/EARIndicator";
import { StatsPanel } from "@/components/gauges/StatsPanel";
import { SessionTimeline } from "@/components/charts/SessionTimeline";
import { AlertBanner } from "@/components/alerts/AlertBanner";
import { AlertFeed } from "@/components/alerts/AlertFeed";

/**
 * Strict session state machine:
 *   idle → running → results
 *
 * Backend enforces: CALIBRATING → MONITORING → ENDED
 * Frontend has a client-side safety timer as fallback.
 */
type PagePhase = "idle" | "running" | "results";

const CLIENT_TIMEOUT_S = 60;

/* ── Animated SVG icons ── */
function AnimatedCheck() {
  return (
    <svg width="120" height="120" viewBox="0 0 120 120" fill="none">
      <motion.circle cx="60" cy="60" r="50" stroke="#4ade80" strokeWidth="5" fill="none"
        initial={{ pathLength: 0 }} animate={{ pathLength: 1 }}
        transition={{ duration: 0.6, ease: "easeOut" }} />
      <motion.path d="M35 60 L52 77 L85 44" stroke="#4ade80" strokeWidth="6" strokeLinecap="round" strokeLinejoin="round" fill="none"
        initial={{ pathLength: 0 }} animate={{ pathLength: 1 }}
        transition={{ duration: 0.4, delay: 0.5, ease: "easeOut" }} />
    </svg>
  );
}

function AnimatedCross() {
  return (
    <svg width="120" height="120" viewBox="0 0 120 120" fill="none">
      <motion.circle cx="60" cy="60" r="50" stroke="#f87171" strokeWidth="5" fill="none"
        initial={{ pathLength: 0 }} animate={{ pathLength: 1 }}
        transition={{ duration: 0.6, ease: "easeOut" }} />
      <motion.path d="M42 42 L78 78" stroke="#f87171" strokeWidth="6" strokeLinecap="round" fill="none"
        initial={{ pathLength: 0 }} animate={{ pathLength: 1 }}
        transition={{ duration: 0.3, delay: 0.5, ease: "easeOut" }} />
      <motion.path d="M78 42 L42 78" stroke="#f87171" strokeWidth="6" strokeLinecap="round" fill="none"
        initial={{ pathLength: 0 }} animate={{ pathLength: 1 }}
        transition={{ duration: 0.3, delay: 0.65, ease: "easeOut" }} />
    </svg>
  );
}

function MonitorPage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [phase, setPhase] = useState<PagePhase>("idle");
  const [startError, setStartError] = useState<string | null>(null);
  const [summary, setSummary] = useState<string | null>(null);
  const stoppingRef = useRef(false);
  const calDoneAt = useRef<number | null>(null);
  const [elapsed, setElapsed] = useState(0);

  const sessionId = useMonitorStore((s) => s.sessionId);
  const setSessionId = useMonitorStore((s) => s.setSessionId);
  const isConnected = useMonitorStore((s) => s.isConnected);
  const currentFrame = useMonitorStore((s) => s.currentFrame);
  const earHistory = useMonitorStore((s) => s.earHistory);
  const fatigueHistory = useMonitorStore((s) => s.fatigueHistory);
  const resetMonitor = useMonitorStore((s) => s.reset);
  const clearAlerts = useAlertStore((s) => s.clearHistory);
  const dismissAlert = useAlertStore((s) => s.dismissAlert);
  const audioEnabled = useSettingsStore((s) => s.audioEnabled);
  const setAudioEnabled = useSettingsStore((s) => s.setAudioEnabled);

  const { startCamera, stopCamera, devices, selectDevice, error: cameraError } =
    useMediaDevices(videoRef);
  const { sendFrame, disconnect } = useWebSocket(sessionId);

  const onFrame = useCallback((blob: Blob) => sendFrame(blob), [sendFrame]);
  useFrameCapture(videoRef, onFrame, phase === "running" && isConnected);
  useAudioAlert();

  const isCalibrating = currentFrame?.is_calibrating ?? true;

  // Track calibration completion
  useEffect(() => {
    if (phase !== "running") return;
    if (!isCalibrating && calDoneAt.current === null) {
      calDoneAt.current = Date.now();
    }
  }, [isCalibrating, phase]);

  // Watch for backend auto_stop
  useEffect(() => {
    if (!currentFrame?.auto_stop || phase !== "running" || stoppingRef.current) return;
    stoppingRef.current = true;
    endSession(currentFrame.session_summary ?? "Session complete.");
  }, [currentFrame?.auto_stop]);

  // Client-side countdown + safety timeout
  useEffect(() => {
    if (phase !== "running") return;
    const tick = setInterval(() => {
      if (calDoneAt.current === null) { setElapsed(0); return; }
      const secs = Math.round((Date.now() - calDoneAt.current) / 1000);
      setElapsed(secs);
      if (secs >= CLIENT_TIMEOUT_S && !stoppingRef.current) {
        stoppingRef.current = true;
        endSession("Monitoring period complete.");
      }
    }, 500);
    return () => clearInterval(tick);
  }, [phase]);

  // Fallback: WS dropped
  useEffect(() => {
    if (phase !== "running" || stoppingRef.current || isConnected || !sessionId) return;
    const t = setTimeout(() => {
      if (!useMonitorStore.getState().isConnected && !stoppingRef.current) {
        stoppingRef.current = true;
        endSession("Connection lost. Session ended.");
      }
    }, 3000);
    return () => clearTimeout(t);
  }, [isConnected, phase, sessionId]);

  // ── Actions ──

  async function handleStart() {
    setStartError(null);
    setSummary(null);
    stoppingRef.current = false;
    calDoneAt.current = null;
    setElapsed(0);
    clearAlerts();
    try {
      await startCamera();
      const { session_id } = await api.startSession();
      setSessionId(session_id);
      setPhase("running");
    } catch (err) {
      setStartError(err instanceof Error ? err.message : "Failed to start");
      setPhase("idle");
    }
  }

  async function endSession(msg: string) {
    const sid = sessionId;

    // Stop everything immediately
    disconnect();
    stopCamera();
    dismissAlert();

    // Transition UI
    setSummary(msg);
    setPhase("results");
    setSessionId(null);

    // Notify backend
    if (sid) {
      try { await api.stopSession(sid); } catch { /* ok */ }
    }
  }

  function handleNewSession() {
    setSummary(null);
    stoppingRef.current = false;
    calDoneAt.current = null;
    setElapsed(0);
    resetMonitor();
    clearAlerts();
    setPhase("idle");
  }

  const isLocked = currentFrame?.is_locked ?? false;
  const calibrationProgress = currentFrame?.calibration_progress ?? 0;

  /* ══════════════════════════════════════════════════
     IDLE — Start screen
  ══════════════════════════════════════════════════ */
  if (phase === "idle") {
    return (
      <div className="flex flex-col items-center gap-10 py-16">
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="text-center">
          <h1 className="text-4xl font-black tracking-tight text-text">Monitoring Dashboard</h1>
          <p className="mt-3 text-text-secondary">Real-time fatigue detection powered by computer vision &amp; ML</p>
        </motion.div>

        <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.15 }} className="flex flex-col items-center gap-5">
          <motion.button whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.97 }} onClick={handleStart}
            className="rounded-2xl bg-primary px-12 py-4 text-base font-bold text-white shadow-lg">
            Start Monitoring
          </motion.button>

          {devices.length > 1 && (
            <select onChange={(e) => selectDevice(e.target.value)}
              className="rounded-xl border border-border bg-surface-raised px-4 py-2.5 text-sm text-text">
              {devices.map((d) => (
                <option key={d.deviceId} value={d.deviceId}>{d.label || `Camera ${d.deviceId.slice(0, 8)}`}</option>
              ))}
            </select>
          )}

          <label className="flex items-center gap-2.5 text-sm text-text-secondary">
            <input type="checkbox" checked={audioEnabled} onChange={(e) => setAudioEnabled(e.target.checked)}
              className="h-4 w-4 rounded border-border accent-primary" />
            Audio alerts enabled
          </label>
        </motion.div>

        {(startError || cameraError) && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
            className="rounded-xl bg-alert-red/10 px-6 py-3 text-sm text-alert-red">
            {startError || cameraError}
          </motion.div>
        )}

        <div className="mt-2 grid max-w-2xl grid-cols-1 gap-3 sm:grid-cols-2">
          {[
            { icon: "👁", title: "Face Detection", desc: "478-point landmark tracking via MediaPipe" },
            { icon: "📊", title: "Fatigue Scoring", desc: "Composite score from PERCLOS, EAR, blinks" },
            { icon: "🧠", title: "Emotion Analysis", desc: "DINOv2-powered expression recognition" },
            { icon: "🔒", title: "Safety Lock", desc: "Auto-stops session on fatigue detection" },
          ].map((f, i) => (
            <motion.div key={f.title} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 + i * 0.08 }} whileHover={{ y: -2 }}
              className="rounded-2xl border border-border bg-surface-raised p-5">
              <span className="text-2xl">{f.icon}</span>
              <h3 className="mt-2 text-sm font-bold text-text">{f.title}</h3>
              <p className="mt-1 text-xs text-text-muted">{f.desc}</p>
            </motion.div>
          ))}
        </div>
      </div>
    );
  }

  /* ══════════════════════════════════════════════════
     RESULTS — Camera off, show results card
  ══════════════════════════════════════════════════ */
  if (phase === "results") {
    const isManual = summary?.includes("manually");
    const hasFatigue = summary?.includes("Fatigue was detected");
    const isSuccess = !hasFatigue && !isManual;

    return (
      <div className="flex flex-col items-center gap-6 py-10">
        {/* Results card — replaces the entire video area */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ type: "spring", stiffness: 200, damping: 20 }}
          className={`w-full max-w-2xl rounded-3xl border-2 p-10 text-center shadow-xl ${
            hasFatigue
              ? "border-red-400/40 bg-linear-to-b from-red-500/5 to-red-500/10"
              : isManual
                ? "border-border bg-surface-raised"
                : "border-green-400/40 bg-linear-to-b from-green-500/5 to-green-500/10"
          }`}
        >
          {/* Animated icon */}
          <div className="flex justify-center mb-4">
            {hasFatigue ? <AnimatedCross /> : isManual ? (
              <motion.div initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ type: "spring", delay: 0.2 }}>
                <span className="text-7xl">⏹️</span>
              </motion.div>
            ) : <AnimatedCheck />}
          </div>

          <motion.h2 initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.6 }}
            className="text-3xl font-black text-text">
            {hasFatigue ? "Fatigue Detected" : isManual ? "Session Stopped" : "All Clear!"}
          </motion.h2>

          <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.8 }}
            className={`mt-2 text-base font-semibold ${
              hasFatigue ? "text-red-400" : isSuccess ? "text-green-400" : "text-text-secondary"
            }`}>
            {hasFatigue
              ? "Drowsiness was detected during your monitoring session"
              : isManual
                ? "You stopped the session manually"
                : "No signs of fatigue detected — you're in great shape!"}
          </motion.p>

          {/* Stats */}
          <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 1.0 }}
            className="mt-6 rounded-2xl border border-border bg-surface-overlay p-5 text-left">
            <p className="text-xs font-bold uppercase tracking-widest text-text-muted mb-3">Session Report</p>
            <p className="text-sm text-text-secondary leading-relaxed">{summary}</p>
          </motion.div>

          {hasFatigue && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1.2 }}
              className="mt-4 rounded-2xl border border-amber-400/30 bg-amber-500/5 p-4">
              <p className="text-xs font-bold uppercase tracking-widest text-amber-400">Recommendation</p>
              <p className="mt-2 text-sm text-text">
                Take a <strong>15-minute break</strong> before your next session. Rest your eyes, stretch, and hydrate.
              </p>
            </motion.div>
          )}
        </motion.div>

        {/* Action buttons */}
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1.3 }}
          className="flex gap-4">
          <motion.button whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.97 }} onClick={handleNewSession}
            className="rounded-2xl bg-primary px-10 py-3.5 text-base font-bold text-white shadow-lg">
            Start New Session
          </motion.button>
          <motion.button whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.97 }}
            onClick={() => { handleNewSession(); window.location.href = "/history"; }}
            className="rounded-2xl border border-border bg-surface-raised px-8 py-3.5 text-base font-medium text-text">
            View History
          </motion.button>
        </motion.div>
      </div>
    );
  }

  /* ══════════════════════════════════════════════════
     RUNNING — Active dashboard
  ══════════════════════════════════════════════════ */
  return (
    <>
      <AlertBanner />
      <div className="flex flex-col gap-4">
        {/* Top bar */}
        <div className="flex items-center justify-between">
          {isLocked ? (
            <div className="flex items-center gap-2 rounded-xl bg-alert-red/10 px-6 py-2.5 text-sm font-bold text-alert-red">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
                <path d="M7 11V7a5 5 0 0 1 10 0v4" />
              </svg>
              Session Locked
            </div>
          ) : (
            <motion.button whileHover={{ scale: 1.03 }} whileTap={{ scale: 0.97 }}
              onClick={() => endSession("Session manually stopped.")}
              className="rounded-xl bg-alert-red px-6 py-2.5 text-sm font-bold text-white shadow-lg">
              Stop Session
            </motion.button>
          )}

          <div className="flex items-center gap-4">
            {isCalibrating ? (
              <span className="rounded-lg bg-amber-500/10 px-3 py-1 text-xs font-bold text-amber-500 animate-pulse">
                Calibrating... {Math.round(calibrationProgress * 100)}%
              </span>
            ) : (
              <span className="rounded-lg bg-surface-overlay px-3 py-1 font-mono text-xs text-text-muted">
                Monitoring: {elapsed}s / {CLIENT_TIMEOUT_S}s
              </span>
            )}
            <label className="flex items-center gap-2 text-xs text-text-muted">
              <input type="checkbox" checked={audioEnabled} onChange={(e) => setAudioEnabled(e.target.checked)}
                className="h-3.5 w-3.5 rounded accent-primary" />
              Audio
            </label>
          </div>
        </div>

        {/* Dashboard */}
        <div className="grid gap-4 lg:grid-cols-[1fr_360px]">
          <VideoFeed videoRef={videoRef} frame={currentFrame} isConnected={isConnected}
            isCalibrating={isCalibrating} calibrationProgress={calibrationProgress} />
          <div className="flex flex-col gap-3">
            <FatigueGauge score={currentFrame?.fatigue_score ?? 0} />
            <StatsPanel frame={currentFrame} />
            <EARIndicator earSmoothed={currentFrame?.ear_smoothed ?? null} threshold={0.2}
              isCalibrating={isCalibrating} />
          </div>
        </div>
        <SessionTimeline earHistory={earHistory} fatigueHistory={fatigueHistory} />
        <AlertFeed />
      </div>
    </>
  );
}

export const monitorRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/monitor",
  component: MonitorPage,
});
