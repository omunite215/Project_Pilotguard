import { useCallback, useEffect, useRef, useState } from "react";
import { useSettingsStore } from "@/stores/settingsStore";

interface MediaDevicesState {
  stream: MediaStream | null;
  devices: MediaDeviceInfo[];
  error: string | null;
  isLoading: boolean;
}

/**
 * Manages camera access and device enumeration.
 * Attaches the selected camera stream to a <video> ref.
 */
export function useMediaDevices(videoRef: React.RefObject<HTMLVideoElement | null>) {
  const [state, setState] = useState<MediaDevicesState>({
    stream: null,
    devices: [],
    error: null,
    isLoading: false,
  });

  const cameraDeviceId = useSettingsStore((s) => s.cameraDeviceId);
  const setCameraDeviceId = useSettingsStore((s) => s.setCameraDeviceId);
  const streamRef = useRef<MediaStream | null>(null);

  const startCamera = useCallback(async (deviceId?: string) => {
    setState((s) => ({ ...s, isLoading: true, error: null }));

    // Stop existing stream
    streamRef.current?.getTracks().forEach((t) => t.stop());

    try {
      const constraints: MediaStreamConstraints = {
        video: deviceId
          ? { deviceId: { exact: deviceId }, width: 640, height: 480 }
          : { width: 640, height: 480, facingMode: "user" },
        audio: false,
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;

      // Attach to video element if it exists now, otherwise it will
      // be attached via the effect below when the ref becomes available
      attachStream(stream);

      // Enumerate devices after permission is granted
      const allDevices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = allDevices.filter((d) => d.kind === "videoinput");

      setState({
        stream,
        devices: videoDevices,
        error: null,
        isLoading: false,
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : "Camera access denied";
      setState((s) => ({ ...s, error: message, isLoading: false }));
    }
  }, [videoRef]);

  function attachStream(stream: MediaStream) {
    const video = videoRef.current;
    if (video) {
      video.srcObject = stream;
      video.play().catch(() => {
        // autoplay may be blocked; user interaction will unblock
      });
    }
  }

  // When the video ref becomes available (after DOM mount), attach the stream
  useEffect(() => {
    if (streamRef.current && videoRef.current && !videoRef.current.srcObject) {
      attachStream(streamRef.current);
    }
  });

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setState((s) => ({ ...s, stream: null }));
  }, [videoRef]);

  const selectDevice = useCallback(
    (deviceId: string) => {
      setCameraDeviceId(deviceId);
      startCamera(deviceId);
    },
    [setCameraDeviceId, startCamera],
  );

  useEffect(() => {
    return () => {
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  return {
    ...state,
    startCamera: () => startCamera(cameraDeviceId || undefined),
    stopCamera,
    selectDevice,
  };
}
