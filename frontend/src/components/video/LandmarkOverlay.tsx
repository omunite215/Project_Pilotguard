import { useEffect, useRef } from "react";
import type { FrameResult } from "@/types/monitoring";

interface LandmarkOverlayProps {
  frame: FrameResult | null;
  containerRef: React.RefObject<HTMLDivElement | null>;
}

// MediaPipe FaceMesh key contour indices (478-point layout)
// Right eye contour
const RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246];
// Left eye contour
const LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398];
// Lips outer
const LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185];
// Lips inner
const LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191];
// Right eyebrow
const RIGHT_BROW = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107];
// Left eyebrow
const LEFT_BROW = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336];
// Face oval
const FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109];
// Nose bridge
const NOSE = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2];

const STATE_COLORS: Record<string, { point: string; contour: string }> = {
  alert:      { point: "#22c55e", contour: "#22c55e" },
  drowsy:     { point: "#f59e0b", contour: "#f59e0b" },
  microsleep: { point: "#ef4444", contour: "#ef4444" },
  unknown:    { point: "#94a3b8", contour: "#94a3b8" },
};

/**
 * Renders all 478 MediaPipe FaceMesh landmarks with key contours.
 * Uses container's rendered size for pixel-perfect alignment.
 */
export function LandmarkOverlay({ frame, containerRef }: LandmarkOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const rect = container.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;
    const dpr = window.devicePixelRatio || 1;

    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    if (!frame?.face_detected || !frame.landmarks) return;

    const pts = frame.landmarks;
    const colors = STATE_COLORS[frame.state] ?? STATE_COLORS.unknown;

    // Draw all 478 points (tiny dots)
    ctx.fillStyle = colors.point;
    ctx.globalAlpha = 0.4;
    for (const [x, y] of pts) {
      ctx.beginPath();
      ctx.arc(x * w, y * h, 1, 0, Math.PI * 2);
      ctx.fill();
    }

    // Draw key contours with higher opacity
    ctx.globalAlpha = 0.9;
    ctx.strokeStyle = colors.contour;
    ctx.lineWidth = 1.5;

    drawContour(ctx, pts, RIGHT_EYE, w, h, true);
    drawContour(ctx, pts, LEFT_EYE, w, h, true);
    drawContour(ctx, pts, LIPS_OUTER, w, h, true);
    drawContour(ctx, pts, RIGHT_BROW, w, h, false);
    drawContour(ctx, pts, LEFT_BROW, w, h, false);
    drawContour(ctx, pts, NOSE, w, h, false);

    // Inner lips slightly thinner
    ctx.lineWidth = 1;
    ctx.globalAlpha = 0.6;
    drawContour(ctx, pts, LIPS_INNER, w, h, true);

    // Face oval very subtle
    ctx.lineWidth = 0.8;
    ctx.globalAlpha = 0.25;
    drawContour(ctx, pts, FACE_OVAL, w, h, true);

    // Eye center dots (bright, for focus point)
    ctx.globalAlpha = 1;
    ctx.fillStyle = colors.contour;
    for (const idx of [468, 473]) { // MediaPipe iris center points
      if (idx < pts.length) {
        const [x, y] = pts[idx];
        ctx.beginPath();
        ctx.arc(x * w, y * h, 3, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    ctx.globalAlpha = 1;
  }, [frame, containerRef]);

  return <canvas ref={canvasRef} className="pointer-events-none absolute inset-0" />;
}

function drawContour(
  ctx: CanvasRenderingContext2D,
  pts: [number, number][],
  indices: number[],
  w: number, h: number,
  close: boolean,
) {
  const valid = indices.filter((i) => i < pts.length);
  if (valid.length < 2) return;
  ctx.beginPath();
  ctx.moveTo(pts[valid[0]][0] * w, pts[valid[0]][1] * h);
  for (let i = 1; i < valid.length; i++) {
    ctx.lineTo(pts[valid[i]][0] * w, pts[valid[i]][1] * h);
  }
  if (close) ctx.closePath();
  ctx.stroke();
}
