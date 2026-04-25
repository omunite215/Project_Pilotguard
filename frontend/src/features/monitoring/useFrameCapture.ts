import { useCallback, useEffect, useRef } from "react";
import { CAPTURE_FPS, JPEG_QUALITY } from "@/lib/constants";
import { useSettingsStore } from "@/stores/settingsStore";

/**
 * Captures video frames from a <video> element at a configurable FPS,
 * encodes them as JPEG blobs, and passes them to onFrame.
 */
export function useFrameCapture(
  videoRef: React.RefObject<HTMLVideoElement | null>,
  onFrame: (blob: Blob) => void,
  enabled: boolean,
) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const rafRef = useRef<number>(0);
  const lastCaptureRef = useRef(0);
  const fps = useSettingsStore((s) => s.captureFps) || CAPTURE_FPS;
  const interval = 1000 / fps;

  const capture = useCallback(() => {
    const video = videoRef.current;
    if (!video || video.readyState < 2 || !enabled) {
      rafRef.current = requestAnimationFrame(capture);
      return;
    }

    const now = performance.now();
    if (now - lastCaptureRef.current < interval) {
      rafRef.current = requestAnimationFrame(capture);
      return;
    }
    lastCaptureRef.current = now;

    if (!canvasRef.current) {
      canvasRef.current = document.createElement("canvas");
    }
    const canvas = canvasRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");
    if (!ctx) {
      rafRef.current = requestAnimationFrame(capture);
      return;
    }

    ctx.drawImage(video, 0, 0);
    canvas.toBlob(
      (blob) => {
        if (blob) onFrame(blob);
      },
      "image/jpeg",
      JPEG_QUALITY,
    );

    rafRef.current = requestAnimationFrame(capture);
  }, [videoRef, onFrame, enabled, interval]);

  useEffect(() => {
    if (enabled) {
      rafRef.current = requestAnimationFrame(capture);
    }
    return () => {
      cancelAnimationFrame(rafRef.current);
    };
  }, [enabled, capture]);
}
