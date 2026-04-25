import { motion, AnimatePresence } from "motion/react";
import { useAlertStore } from "@/stores/alertStore";

const LEVEL_STYLES: Record<string, { border: string; bg: string; text: string; icon: string }> = {
  advisory: { border: "border-advisory/40", bg: "bg-advisory/10", text: "text-advisory", icon: "⚡" },
  caution: { border: "border-caution/40", bg: "bg-caution/10", text: "text-caution", icon: "⚠️" },
  warning: { border: "border-warning/40", bg: "bg-warning/10", text: "text-warning", icon: "🚨" },
};

export function AlertFeed() {
  const alertHistory = useAlertStore((s) => s.alertHistory);
  const clearHistory = useAlertStore((s) => s.clearHistory);

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.3 }}
      className="rounded-2xl border border-border bg-surface-raised p-4 transition-colors"
    >
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-bold text-text">
          Alerts {alertHistory.length > 0 && `(${alertHistory.length})`}
        </h3>
        {alertHistory.length > 0 && (
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={clearHistory}
            className="rounded-lg px-3 py-1 text-xs font-medium text-text-muted transition-colors hover:bg-surface-overlay hover:text-text"
          >
            Clear
          </motion.button>
        )}
      </div>

      {alertHistory.length === 0 ? (
        <div className="flex items-center gap-2 py-3 text-sm text-text-muted">
          <span className="text-alert-green">●</span>
          All clear — no alerts
        </div>
      ) : (
        <div className="flex max-h-52 flex-col gap-2 overflow-y-auto pr-1">
          <AnimatePresence initial={false}>
            {[...alertHistory].reverse().map((alert, i) => {
              const style = LEVEL_STYLES[alert.level] ?? LEVEL_STYLES.advisory;
              return (
                <motion.div
                  key={`${alert.timestamp}-${i}`}
                  initial={{ opacity: 0, x: -20, height: 0 }}
                  animate={{ opacity: 1, x: 0, height: "auto" }}
                  exit={{ opacity: 0, x: 20, height: 0 }}
                  transition={{ type: "spring", stiffness: 300, damping: 25 }}
                  className={`overflow-hidden rounded-xl border-l-4 px-3.5 py-2.5 ${style.border} ${style.bg}`}
                >
                  <div className="flex items-center justify-between">
                    <span className={`text-xs font-black uppercase tracking-wider ${style.text}`}>
                      {style.icon} {alert.level}
                    </span>
                    <span className="font-mono text-[10px] text-text-muted">
                      {new Date(alert.timestamp * 1000).toLocaleTimeString()}
                    </span>
                  </div>
                  <p className="mt-1 text-xs text-text-secondary">{alert.message}</p>
                </motion.div>
              );
            })}
          </AnimatePresence>
        </div>
      )}
    </motion.div>
  );
}
