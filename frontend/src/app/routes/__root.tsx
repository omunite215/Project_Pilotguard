import { createRootRoute, Outlet, Link, useLocation } from "@tanstack/react-router";
import { motion, AnimatePresence } from "motion/react";
import { useEffect } from "react";
import { useMonitorStore } from "@/stores/monitorStore";
import { useSettingsStore } from "@/stores/settingsStore";

/* ── Dark mode wiring ── */
function useDarkMode() {
  const dark = useSettingsStore((s) => s.darkMode);
  const toggle = useSettingsStore((s) => s.setDarkMode);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
  }, [dark]);

  return { dark, toggle: () => toggle(!dark) };
}

/* ── Icons (inline SVG to avoid deps) ── */
function SunIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="5" /><line x1="12" y1="1" x2="12" y2="3" /><line x1="12" y1="21" x2="12" y2="23" /><line x1="4.22" y1="4.22" x2="5.64" y2="5.64" /><line x1="18.36" y1="18.36" x2="19.78" y2="19.78" /><line x1="1" y1="12" x2="3" y2="12" /><line x1="21" y1="12" x2="23" y2="12" /><line x1="4.22" y1="19.78" x2="5.64" y2="18.36" /><line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
    </svg>
  );
}

function MoonIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
    </svg>
  );
}

/* ── Connection badge ── */
function ConnectionBadge() {
  const isConnected = useMonitorStore((s) => s.isConnected);
  const sessionId = useMonitorStore((s) => s.sessionId);
  if (!sessionId) return null;

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      className="flex items-center gap-2"
    >
      <span className={`h-2 w-2 rounded-full ${isConnected ? "bg-alert-green animate-pulse" : "bg-alert-red"}`} />
      <span className="text-xs text-text-muted">
        {isConnected ? "Live" : "Reconnecting..."}
      </span>
    </motion.div>
  );
}

/* ── Nav links ── */
const NAV_LINKS = [
  { to: "/" as const, label: "Home" },
  { to: "/monitor" as const, label: "Monitor" },
  { to: "/history" as const, label: "History" },
] as const;

/* ── Main layout ── */
function RootLayout() {
  const { dark, toggle } = useDarkMode();
  const location = useLocation();

  return (
    <div className="min-h-screen bg-surface-dim text-text transition-colors duration-300">
      {/* ── Header ── */}
      <header className="glass sticky top-0 z-40 border-b border-border">
        <nav className="mx-auto flex h-14 max-w-7xl items-center justify-between px-5">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary text-white text-sm font-black">
              PG
            </div>
            <span className="text-base font-bold tracking-tight text-text">PilotGuard</span>
          </Link>

          {/* Nav links */}
          <div className="flex items-center gap-1 rounded-xl bg-surface-overlay p-1">
            {NAV_LINKS.map((link) => {
              const active = location.pathname === link.to;
              return (
                <Link key={link.to} to={link.to} className="relative px-4 py-1.5 text-sm font-medium transition-colors">
                  {active && (
                    <motion.div
                      layoutId="nav-pill"
                      className="absolute inset-0 rounded-lg bg-primary/10"
                      transition={{ type: "spring", stiffness: 400, damping: 30 }}
                    />
                  )}
                  <span className={`relative z-10 ${active ? "text-primary" : "text-text-secondary hover:text-text"}`}>
                    {link.label}
                  </span>
                </Link>
              );
            })}
          </div>

          {/* Right: connection + theme toggle */}
          <div className="flex items-center gap-3">
            <ConnectionBadge />
            <motion.button
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              onClick={toggle}
              className="flex h-9 w-9 items-center justify-center rounded-xl bg-surface-overlay text-text-secondary transition-colors hover:text-text"
              title={dark ? "Switch to light mode" : "Switch to dark mode"}
            >
              <AnimatePresence mode="wait">
                <motion.div
                  key={dark ? "moon" : "sun"}
                  initial={{ rotate: -90, opacity: 0 }}
                  animate={{ rotate: 0, opacity: 1 }}
                  exit={{ rotate: 90, opacity: 0 }}
                  transition={{ duration: 0.2 }}
                >
                  {dark ? <SunIcon /> : <MoonIcon />}
                </motion.div>
              </AnimatePresence>
            </motion.button>
          </div>
        </nav>
      </header>

      {/* ── Page content with route transition ── */}
      <main className="mx-auto max-w-7xl px-5 py-5">
        <AnimatePresence mode="wait">
          <motion.div
            key={location.pathname}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -12 }}
            transition={{ duration: 0.2, ease: "easeOut" }}
          >
            <Outlet />
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  );
}

function RootErrorBoundary({ error }: { error: Error }) {
  return (
    <div className="flex min-h-screen items-center justify-center bg-surface-dim">
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        className="glass rounded-2xl border border-border p-8 shadow-xl"
      >
        <h1 className="mb-2 text-2xl font-bold text-warning">Something went wrong</h1>
        <p className="text-text-secondary">{error.message}</p>
      </motion.div>
    </div>
  );
}

export const rootRoute = createRootRoute({
  component: RootLayout,
  errorComponent: RootErrorBoundary,
});
