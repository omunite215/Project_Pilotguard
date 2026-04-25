import { useCallback, useEffect, useRef } from "react";
import { useMonitorStore } from "@/stores/monitorStore";
import { useAlertStore } from "@/stores/alertStore";
import { wsUrl, RECONNECT_BASE_DELAY_MS, RECONNECT_MAX_DELAY_MS } from "@/lib/constants";
import type { FrameResult } from "@/types/monitoring";

/**
 * Manages the WebSocket connection to the backend for real-time frame results.
 * Handles reconnection with exponential backoff.
 */
export function useWebSocket(sessionId: string | null) {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttempt = useRef(0);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const sessionIdRef = useRef(sessionId);
  sessionIdRef.current = sessionId;

  // Flag to suppress reconnection after an intentional disconnect
  const intentionalDisconnect = useRef(false);

  // Use refs for store actions to avoid re-creating callbacks
  const setConnected = useMonitorStore((s) => s.setConnected);
  const setCurrentFrame = useMonitorStore((s) => s.setCurrentFrame);
  const pushAlert = useAlertStore((s) => s.pushAlert);
  const storeRef = useRef({ setConnected, setCurrentFrame, pushAlert });
  storeRef.current = { setConnected, setCurrentFrame, pushAlert };

  const connect = useCallback(() => {
    const sid = sessionIdRef.current;
    if (!sid) return;

    // Don't open a new connection if one is already open/connecting
    if (wsRef.current?.readyState === WebSocket.OPEN || wsRef.current?.readyState === WebSocket.CONNECTING) {
      return;
    }

    intentionalDisconnect.current = false;

    const ws = new WebSocket(wsUrl(sid));
    wsRef.current = ws;

    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
      storeRef.current.setConnected(true);
      reconnectAttempt.current = 0;
    };

    ws.onmessage = (event: MessageEvent) => {
      try {
        const text = typeof event.data === "string" ? event.data : new TextDecoder().decode(event.data);
        const frame = JSON.parse(text) as FrameResult;
        storeRef.current.setCurrentFrame(frame);
        if (frame.alert) {
          storeRef.current.pushAlert(frame.alert);
        }
      } catch {
        // Ignore malformed messages
      }
    };

    ws.onclose = () => {
      storeRef.current.setConnected(false);
      wsRef.current = null;

      // Only reconnect if session is still active AND we didn't intentionally disconnect
      if (sessionIdRef.current && !intentionalDisconnect.current) {
        const delay = Math.min(
          RECONNECT_BASE_DELAY_MS * 2 ** reconnectAttempt.current,
          RECONNECT_MAX_DELAY_MS,
        );
        reconnectAttempt.current += 1;
        reconnectTimer.current = setTimeout(connect, delay);
      }
    };

    ws.onerror = () => {
      // onclose will fire after this
    };
  }, []); // No deps — uses refs for everything

  const sendFrame = useCallback((blob: Blob) => {
    const ws = wsRef.current;
    if (ws?.readyState === WebSocket.OPEN) {
      blob.arrayBuffer().then((buf) => ws.send(buf));
    }
  }, []);

  const disconnect = useCallback(() => {
    clearTimeout(reconnectTimer.current);
    reconnectAttempt.current = 0;
    intentionalDisconnect.current = true;
    const ws = wsRef.current;
    wsRef.current = null;
    ws?.close();
    storeRef.current.setConnected(false);
  }, []);

  // Connect when sessionId changes, disconnect on cleanup
  useEffect(() => {
    if (sessionId) {
      connect();
    } else {
      disconnect();
    }
    return () => {
      disconnect();
    };
  }, [sessionId, connect, disconnect]);

  return { sendFrame, disconnect };
}
