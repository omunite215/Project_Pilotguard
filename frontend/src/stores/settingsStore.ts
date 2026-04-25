import { create } from "zustand";
import { persist } from "zustand/middleware";

interface SettingsState {
  cameraDeviceId: string;
  captureFps: number;
  alertThresholds: {
    advisory: number;
    caution: number;
    warning: number;
  };
  audioEnabled: boolean;
  darkMode: boolean;

  setCameraDeviceId: (id: string) => void;
  setCaptureFps: (fps: number) => void;
  setAudioEnabled: (enabled: boolean) => void;
  setDarkMode: (dark: boolean) => void;
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set) => ({
      cameraDeviceId: "",
      captureFps: 15,
      alertThresholds: {
        advisory: 30,
        caution: 55,
        warning: 75,
      },
      audioEnabled: true,
      darkMode: false,

      setCameraDeviceId: (id) => set({ cameraDeviceId: id }),
      setCaptureFps: (fps) => set({ captureFps: fps }),
      setAudioEnabled: (enabled) => set({ audioEnabled: enabled }),
      setDarkMode: (dark) => set({ darkMode: dark }),
    }),
    { name: "pilotguard-settings" },
  ),
);
