import { useRef } from "react";
import { motion, AnimatePresence } from "motion/react";
import type { FrameResult } from "@/types/monitoring";
import { LandmarkOverlay } from "./LandmarkOverlay";

interface VideoFeedProps {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  frame: FrameResult | null;
  isConnected: boolean;
  isCalibrating: boolean;
  calibrationProgress: number;
}

const STATE_BANNER: Record<string, { text: string; bg: string; glow: string }> = {
  alert: { text: "ALERT — You're doing great!", bg: "bg-alert-green/90", glow: "glow-green" },
  drowsy: { text: "DROWSY — Stay focused!", bg: "bg-alert-amber/90", glow: "glow-amber" },
  microsleep: { text: "MICROSLEEP — Wake up!", bg: "bg-alert-red/90 animate-pulse", glow: "glow-red" },
  unknown: { text: "Detecting...", bg: "bg-text-muted/60", glow: "" },
};

export function VideoFeed({
  videoRef, frame, isConnected, isCalibrating, calibrationProgress,
}: VideoFeedProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const banner = STATE_BANNER[frame?.state ?? "unknown"] ?? STATE_BANNER.unknown;

  return (
    <motion.div
      ref={containerRef}
      initial={{ opacity: 0, scale: 0.97 }}
      animate={{ opacity: 1, scale: 1 }}
      className={`relative overflow-hidden rounded-2xl bg-surface-dim shadow-xl ring-1 ring-border ${banner.glow}`}
    >
      <video
        ref={videoRef}
        autoPlay playsInline muted
        className="block h-full w-full object-cover"
      />

      {/* Landmark overlay — uses container size for correct positioning */}
      <LandmarkOverlay frame={frame} containerRef={containerRef} />

      {/* Top-right badges */}
      <div className="absolute right-3 top-3 flex flex-col items-end gap-2">
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          className={`flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-bold text-white shadow-lg backdrop-blur-sm ${
            isConnected ? "bg-alert-green/80" : "bg-alert-red/80"
          }`}
        >
          <motion.span
            animate={{ scale: isConnected ? [1, 1.3, 1] : 1 }}
            transition={{ repeat: Infinity, duration: 2 }}
            className={`h-1.5 w-1.5 rounded-full ${isConnected ? "bg-white" : "bg-white/50"}`}
          />
          {isConnected ? "Live" : "Offline"}
        </motion.div>

        {frame?.processing_time_ms != null && (
          <div className="rounded-full bg-black/50 px-2.5 py-1 font-mono text-[10px] text-white/80 backdrop-blur-sm">
            {frame.processing_time_ms.toFixed(0)}ms
          </div>
        )}
      </div>

      {/* Emotion badge top-left */}
      <AnimatePresence>
        {frame?.emotion && frame.emotion !== "unknown" && (
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            className="absolute left-3 top-3 rounded-full bg-primary/80 px-3 py-1.5 text-xs font-bold text-white shadow-lg backdrop-blur-sm"
          >
            {frame.emotion} {((frame.emotion_confidence ?? 0) * 100).toFixed(0)}%
          </motion.div>
        )}
      </AnimatePresence>

      {/* Bottom state banner */}
      <AnimatePresence>
        {frame && !isCalibrating && (
          <motion.div
            key={frame.state}
            initial={{ y: 40, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            exit={{ y: 40, opacity: 0 }}
            transition={{ type: "spring", stiffness: 300, damping: 25 }}
            className={`absolute inset-x-0 bottom-0 py-2.5 text-center text-sm font-black tracking-wide text-white ${banner.bg}`}
          >
            {banner.text}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Calibration overlay */}
      <AnimatePresence>
        {isCalibrating && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 flex flex-col items-center justify-center bg-black/70 backdrop-blur-sm"
          >
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ repeat: Infinity, duration: 1.5, ease: "linear" }}
              className="mb-5 h-16 w-16 rounded-full border-4 border-white/20 border-t-primary-light"
            />
            <p className="text-lg font-bold text-white">Calibrating Baseline</p>
            <div className="mt-4 h-2 w-56 overflow-hidden rounded-full bg-white/20">
              <motion.div
                className="h-full rounded-full bg-linear-to-r from-primary-light to-primary"
                animate={{ width: `${calibrationProgress * 100}%` }}
                transition={{ duration: 0.3 }}
              />
            </div>
            <p className="mt-3 text-sm text-white/60">
              Look straight ahead — {(calibrationProgress * 100).toFixed(0)}%
            </p>
          </motion.div>
        )}
      </AnimatePresence>

      {/* No face overlay */}
      <AnimatePresence>
        {!isCalibrating && frame && !frame.face_detected && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 flex items-center justify-center bg-black/50 backdrop-blur-sm"
          >
            <motion.div
              animate={{ scale: [1, 1.03, 1] }}
              transition={{ repeat: Infinity, duration: 2 }}
              className="rounded-2xl bg-alert-amber/90 px-8 py-4 text-center shadow-2xl"
            >
              <p className="text-lg font-black text-white">No Face Detected</p>
              <p className="mt-1 text-sm text-white/80">Position your face in view</p>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
