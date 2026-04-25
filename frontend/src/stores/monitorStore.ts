import { create } from "zustand";
import type { FrameResult } from "@/types/monitoring";

interface MonitorState {
  isConnected: boolean;
  currentFrame: FrameResult | null;
  sessionId: string | null;
  earHistory: Array<{ t: number; ear: number }>;
  fatigueHistory: Array<{ t: number; score: number }>;

  setConnected: (connected: boolean) => void;
  setCurrentFrame: (frame: FrameResult) => void;
  setSessionId: (id: string | null) => void;
  reset: () => void;
}

export const useMonitorStore = create<MonitorState>((set) => ({
  isConnected: false,
  currentFrame: null,
  sessionId: null,
  earHistory: [],
  fatigueHistory: [],

  setConnected: (connected) => set({ isConnected: connected }),

  setCurrentFrame: (frame) =>
    set((state) => ({
      currentFrame: frame,
      earHistory: [
        ...state.earHistory.slice(-1800),
        { t: frame.timestamp, ear: frame.ear_smoothed ?? 0 },
      ],
      fatigueHistory: [
        ...state.fatigueHistory.slice(-1800),
        { t: frame.timestamp, score: frame.fatigue_score },
      ],
    })),

  setSessionId: (id) => set({ sessionId: id }),

  reset: () =>
    set({
      isConnected: false,
      currentFrame: null,
      sessionId: null,
      earHistory: [],
      fatigueHistory: [],
    }),
}));
