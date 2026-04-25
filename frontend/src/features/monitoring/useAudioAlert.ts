import { useEffect, useRef } from "react";
import { useAlertStore } from "@/stores/alertStore";
import { useMonitorStore } from "@/stores/monitorStore";
import { useSettingsStore } from "@/stores/settingsStore";

/**
 * Safety-critical audio alert system.
 *
 * During LOCKED state:
 *   - failure.mp3 loops continuously (cannot be silenced)
 *   - Speech announces the lock status
 *
 * On unlock:
 *   - failure.mp3 stops
 *   - success.mp3 plays
 *   - "All clear" speech
 */
export function useAudioAlert() {
  const audioEnabled = useSettingsStore((s) => s.audioEnabled);
  const activeAlert = useAlertStore((s) => s.activeAlert);
  const frame = useMonitorStore((s) => s.currentFrame);

  const lastAlertRef = useRef<number>(0);
  const wasLockedRef = useRef(false);
  const failureAudio = useRef<HTMLAudioElement | null>(null);
  const successAudio = useRef<HTMLAudioElement | null>(null);
  const loopRef = useRef<ReturnType<typeof setInterval> | undefined>(undefined);
  const speechCooldown = useRef(0);

  const isLocked = frame?.is_locked ?? false;

  // Pre-load audio
  useEffect(() => {
    failureAudio.current = new Audio("/failure.mp3");
    failureAudio.current.volume = 0.6;
    successAudio.current = new Audio("/success.mp3");
    successAudio.current.volume = 0.5;
  }, []);

  function playFailure() {
    const audio = failureAudio.current;
    if (audio) {
      audio.currentTime = 0;
      audio.play().catch(() => {});
    }
  }

  function playSuccess() {
    const audio = successAudio.current;
    if (audio) {
      audio.currentTime = 0;
      audio.play().catch(() => {});
    }
  }

  function stopFailure() {
    const audio = failureAudio.current;
    if (audio) {
      audio.pause();
      audio.currentTime = 0;
    }
  }

  function startAlarmLoop() {
    stopAlarmLoop();
    playFailure();
    loopRef.current = setInterval(playFailure, 5000);
  }

  function stopAlarmLoop() {
    if (loopRef.current !== undefined) {
      clearInterval(loopRef.current);
      loopRef.current = undefined;
    }
    stopFailure();
  }

  function speak(text: string) {
    const now = Date.now();
    // Rate-limit speech to once per 8 seconds
    if (now - speechCooldown.current < 8000) return;
    speechCooldown.current = now;

    try {
      if ("speechSynthesis" in window) {
        window.speechSynthesis.cancel();
        const utt = new SpeechSynthesisUtterance(text);
        utt.rate = 1.0;
        utt.pitch = 1.0;
        utt.volume = 0.8;
        window.speechSynthesis.speak(utt);
      }
    } catch {
      // Not available
    }
  }

  // Handle lock state changes
  useEffect(() => {
    if (!audioEnabled) {
      stopAlarmLoop();
      return;
    }

    if (isLocked) {
      // Locked — sustain alarm
      if (!wasLockedRef.current) {
        // Just entered lock
        startAlarmLoop();

        const level = frame?.lock_level ?? "warning";
        if (level === "microsleep") {
          speak("Microsleep detected! System locked. You must demonstrate sustained alertness to unlock.");
        } else if (level === "warning") {
          speak("Critical fatigue detected! System locked. Stop and rest immediately.");
        } else if (level === "caution") {
          speak("Caution! Fatigue detected. System locked. Take a break.");
        } else {
          speak("Advisory. Fatigue signs detected. System monitoring locked.");
        }
      }
      wasLockedRef.current = true;
    } else {
      // Not locked
      if (wasLockedRef.current) {
        // Just unlocked — play success
        stopAlarmLoop();
        // Only speak if session is not auto-stopping (auto_stop means session is ending)
        if (!frame?.auto_stop) {
          playSuccess();
          speak("All clear. Lock released. You're doing well.");
        }
        wasLockedRef.current = false;
      }
    }
  }, [isLocked, audioEnabled]);

  // Handle new alerts (non-locked initial trigger)
  useEffect(() => {
    if (!audioEnabled || !activeAlert) return;
    if (activeAlert.timestamp === lastAlertRef.current) return;
    lastAlertRef.current = activeAlert.timestamp;

    if (!isLocked) {
      playFailure();
    }
  }, [activeAlert, audioEnabled, isLocked]);

  // Pilot success messages (e.g., "5 minutes of sustained alertness")
  const pilotMessage = frame?.pilot_message ?? null;
  const lastPilotMsg = useRef<string | null>(null);

  useEffect(() => {
    if (!audioEnabled || !pilotMessage || pilotMessage === lastPilotMsg.current) return;
    lastPilotMsg.current = pilotMessage;
    playSuccess();
    speak(pilotMessage);
  }, [pilotMessage, audioEnabled]);

  // Cleanup
  useEffect(() => {
    return () => {
      stopAlarmLoop();
      window.speechSynthesis?.cancel();
    };
  }, []);
}
