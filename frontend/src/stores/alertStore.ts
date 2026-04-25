import { create } from "zustand";
import type { Alert } from "@/types/monitoring";

interface AlertState {
  activeAlert: Alert | null;
  alertHistory: Alert[];
  audioEnabled: boolean;

  pushAlert: (alert: Alert) => void;
  dismissAlert: () => void;
  clearHistory: () => void;
}

export const useAlertStore = create<AlertState>((set) => ({
  activeAlert: null,
  alertHistory: [],
  audioEnabled: true,

  pushAlert: (alert) =>
    set((state) => ({
      activeAlert: alert,
      alertHistory: [...state.alertHistory, alert],
    })),

  dismissAlert: () => set({ activeAlert: null }),

  clearHistory: () => set({ alertHistory: [] }),
}));
