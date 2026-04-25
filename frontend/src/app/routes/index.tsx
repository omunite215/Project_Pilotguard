import { createRoute, Link } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { motion } from "motion/react";
import { rootRoute } from "./__root";
import { api } from "@/lib/api";

function IndexPage() {
  const { data: health } = useQuery({
    queryKey: ["health"],
    queryFn: () => api.health(),
    refetchInterval: 10_000,
  });

  const modelsReady = health?.models_loaded
    ? Object.values(health.models_loaded).filter(Boolean).length
    : 0;
  const totalModels = health?.models_loaded
    ? Object.keys(health.models_loaded).length
    : 0;
  const allReady = modelsReady === totalModels && totalModels > 0;

  return (
    <div className="flex flex-col items-center gap-12 py-20">
      {/* Hero */}
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="text-center"
      >
        <motion.div
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ type: "spring", stiffness: 200, delay: 0.1 }}
          className="mx-auto mb-6 flex h-20 w-20 items-center justify-center rounded-3xl bg-primary text-3xl font-black text-white shadow-xl"
          style={{ boxShadow: "0 8px 40px var(--color-primary-glow)" }}
        >
          PG
        </motion.div>
        <h1 className="text-5xl font-black tracking-tight text-text">
          PilotGuard
        </h1>
        <p className="mt-4 max-w-lg text-lg text-text-secondary">
          Real-time pilot cognitive state monitoring powered by computer vision
          and deep learning
        </p>
      </motion.div>

      {/* System status */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="flex flex-col items-center gap-4"
      >
        <div
          className={`flex items-center gap-2.5 rounded-full border px-5 py-2.5 text-sm font-semibold transition-colors ${
            allReady
              ? "border-alert-green/30 bg-alert-green/10 text-alert-green"
              : "border-alert-amber/30 bg-alert-amber/10 text-alert-amber"
          }`}
        >
          <motion.span
            animate={allReady ? { scale: [1, 1.3, 1] } : {}}
            transition={{ repeat: Infinity, duration: 2 }}
            className={`h-2.5 w-2.5 rounded-full ${allReady ? "bg-alert-green" : "bg-alert-amber"}`}
          />
          {allReady
            ? "All Systems Ready"
            : health
              ? `Loading models (${modelsReady}/${totalModels})`
              : "Connecting to backend..."}
        </div>

        {health?.models_loaded && (
          <div className="flex flex-wrap justify-center gap-2">
            {Object.entries(health.models_loaded).map(([name, loaded], i) => (
              <motion.span
                key={name}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.4 + i * 0.08 }}
                className={`rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors ${
                  loaded
                    ? "border-alert-green/20 bg-alert-green/10 text-alert-green"
                    : "border-border bg-surface-overlay text-text-muted"
                }`}
              >
                {loaded ? "✓" : "○"} {name.replace(/_/g, " ")}
              </motion.span>
            ))}
          </div>
        )}
      </motion.div>

      {/* CTA */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
      >
        <Link to="/monitor">
          <motion.div
            whileHover={{ scale: 1.04, boxShadow: "0 12px 40px var(--color-primary-glow)" }}
            whileTap={{ scale: 0.97 }}
            className="rounded-2xl bg-primary px-12 py-4 text-base font-bold text-white shadow-xl transition-shadow"
          >
            Start Monitoring
          </motion.div>
        </Link>
      </motion.div>

      {/* Tech stack cards */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.6 }}
        className="mt-4 grid max-w-3xl grid-cols-1 gap-4 sm:grid-cols-3"
      >
        {[
          { icon: "👁", title: "Computer Vision", desc: "MediaPipe 478-landmark face detection with Kalman-filtered EAR tracking" },
          { icon: "🧠", title: "Deep Learning", desc: "DINOv2 ViT-S/14 for emotion recognition and drowsiness classification" },
          { icon: "⚡", title: "Real-Time", desc: "Sub-20ms latency pipeline with WebSocket streaming at 15 FPS" },
        ].map((card, i) => (
          <motion.div
            key={card.title}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.7 + i * 0.1 }}
            whileHover={{ y: -4 }}
            className="rounded-2xl border border-border bg-surface-raised p-6 transition-colors"
          >
            <span className="text-3xl">{card.icon}</span>
            <h3 className="mt-3 text-sm font-bold text-text">{card.title}</h3>
            <p className="mt-2 text-xs leading-relaxed text-text-muted">{card.desc}</p>
          </motion.div>
        ))}
      </motion.div>
    </div>
  );
}

export const indexRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/",
  component: IndexPage,
});
