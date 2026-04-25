import { motion, AnimatePresence } from "motion/react";
import { useAlertStore } from "@/stores/alertStore";
import { useMonitorStore } from "@/stores/monitorStore";

const STYLES: Record<string, { bg: string; border: string; icon: string }> = {
  advisory: { bg: "bg-advisory/95", border: "border-advisory", icon: "⚡" },
  caution: { bg: "bg-caution/95", border: "border-caution", icon: "⚠️" },
  warning: { bg: "bg-warning/95", border: "border-warning", icon: "🚨" },
};

/**
 * Full-width alert banner. During lock state, it CANNOT be dismissed
 * and shows a countdown + unlock progress bar.
 */
export function AlertBanner() {
  const activeAlert = useAlertStore((s) => s.activeAlert);
  const dismissAlert = useAlertStore((s) => s.dismissAlert);
  const frame = useMonitorStore((s) => s.currentFrame);

  const isLocked = frame?.is_locked ?? false;
  const lockRemaining = frame?.lock_remaining_seconds ?? 0;
  const lockProgress = frame?.lock_progress ?? 0;

  // Only allow dismiss when NOT locked
  const canDismiss = !isLocked;

  return (
    <AnimatePresence>
      {activeAlert && (
        <motion.div
          initial={{ y: -100, opacity: 0, scale: 0.95 }}
          animate={{ y: 0, opacity: 1, scale: 1 }}
          exit={{ y: -100, opacity: 0, scale: 0.95 }}
          transition={{ type: "spring", stiffness: 400, damping: 28 }}
          className={`fixed inset-x-0 top-0 z-50 border-b-2 shadow-2xl ${STYLES[activeAlert.level]?.bg ?? "bg-advisory/95"} ${STYLES[activeAlert.level]?.border ?? ""}`}
        >
          <div className="mx-auto max-w-7xl px-5 py-3.5">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <motion.span
                  animate={{ scale: [1, 1.2, 1] }}
                  transition={{ repeat: Infinity, duration: 1 }}
                  className="text-2xl"
                >
                  {STYLES[activeAlert.level]?.icon}
                </motion.span>
                <div>
                  <div className="flex items-center gap-3">
                    <p className="text-xs font-black uppercase tracking-widest text-white/80">
                      {activeAlert.level} Alert
                    </p>
                    {isLocked && (
                      <span className="rounded-full bg-white/20 px-2.5 py-0.5 text-[10px] font-bold text-white">
                        LOCKED — {lockRemaining.toFixed(0)}s
                      </span>
                    )}
                  </div>
                  <p className="text-sm font-medium text-white">{activeAlert.message}</p>
                </div>
              </div>

              {canDismiss ? (
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={dismissAlert}
                  className="rounded-xl bg-white/20 px-5 py-2 text-sm font-bold text-white backdrop-blur-sm transition-colors hover:bg-white/30"
                >
                  Dismiss
                </motion.button>
              ) : (
                <div className="flex items-center gap-2 text-xs font-bold text-white/80">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                    <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
                    <path d="M7 11V7a5 5 0 0 1 10 0v4" />
                  </svg>
                  Cannot dismiss
                </div>
              )}
            </div>

            {/* Lock progress bar */}
            {isLocked && (
              <div className="mt-2.5 flex items-center gap-3">
                <div className="flex-1">
                  <div className="h-1.5 overflow-hidden rounded-full bg-white/20">
                    <motion.div
                      className="h-full rounded-full bg-white/60"
                      animate={{ width: `${lockProgress * 100}%` }}
                      transition={{ duration: 0.3 }}
                    />
                  </div>
                </div>
                <span className="text-[10px] font-mono text-white/60">
                  Alertness: {(lockProgress * 100).toFixed(0)}%
                </span>
              </div>
            )}
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
