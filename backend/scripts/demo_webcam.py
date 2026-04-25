"""PilotGuard webcam demo — live landmark detection + EAR overlay.

Usage:
    cd backend
    python -m scripts.demo_webcam

Controls:
    q / ESC  — quit
    c        — restart calibration
    space    — toggle pause
"""

from __future__ import annotations

import sys
import time

import cv2
import numpy as np

from src.cv.pipeline import CognitiveState, CVPipeline

# ── Colors (BGR) ──────────────────────────────────────────────────────────────
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
CYAN = (255, 255, 0)
WHITE = (255, 255, 255)
ORANGE = (0, 165, 255)

STATE_COLORS = {
    CognitiveState.ALERT: GREEN,
    CognitiveState.DROWSY: YELLOW,
    CognitiveState.MICROSLEEP: RED,
    CognitiveState.UNKNOWN: WHITE,
}


def draw_landmarks(
    frame: np.ndarray,
    points: np.ndarray,
    frame_w: int,
    frame_h: int,
) -> None:
    """Draw 68 landmark points on the frame."""
    for x_norm, y_norm in points:
        x = int(x_norm * frame_w)
        y = int(y_norm * frame_h)
        cv2.circle(frame, (x, y), 1, GREEN, -1)


def draw_eye_contour(
    frame: np.ndarray,
    eye_points: np.ndarray,
    frame_w: int,
    frame_h: int,
    color: tuple[int, int, int],
) -> None:
    """Draw eye contour as a polygon."""
    pts = np.array(
        [(int(x * frame_w), int(y * frame_h)) for x, y in eye_points],
        dtype=np.int32,
    )
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=1)


def draw_hud(
    frame: np.ndarray,
    result: object,
    fps: float,
) -> None:
    """Draw heads-up display with metrics."""
    h, w = frame.shape[:2]
    y = 25
    line_h = 22

    def put(text: str, color: tuple[int, int, int] = WHITE) -> None:
        nonlocal y
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y += line_h

    put(f"FPS: {fps:.0f}", CYAN)
    put(f"Processing: {result.processing_time_ms:.1f}ms", CYAN)

    if result.is_calibrating:
        progress = result.calibration_progress * 100
        put(f"CALIBRATING... {progress:.0f}%", ORANGE)
        # Draw calibration progress bar
        bar_x, bar_y, bar_w, bar_h = 10, y, 200, 15
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), WHITE, 1)
        fill_w = int(bar_w * result.calibration_progress)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), ORANGE, -1)
        y += line_h + 5
    else:
        if result.ear:
            put(f"EAR: {result.ear.average:.3f}", WHITE)
        if result.ear_smoothed:
            put(f"EAR (smooth): {result.ear_smoothed.ear_smoothed:.3f}", WHITE)
        if result.mar is not None:
            put(f"MAR: {result.mar:.3f}", WHITE)

        state_color = STATE_COLORS.get(result.state, WHITE)
        put(f"State: {result.state.name}", state_color)
        put(f"PERCLOS: {result.perclos:.1f}%", YELLOW if result.perclos > 25 else WHITE)
        put(f"Blink Rate: {result.blink_rate_pm:.0f}/min", WHITE)
        put(f"Eyes: {'CLOSED' if result.is_eyes_closed else 'OPEN'}", RED if result.is_eyes_closed else GREEN)

    # Blink event notification (flash briefly)
    if result.blink_event:
        evt = result.blink_event
        label = "MICROSLEEP!" if evt.is_microsleep else "Blink"
        color = RED if evt.is_microsleep else YELLOW
        cv2.putText(
            frame,
            f"{label} ({evt.duration_ms:.0f}ms)",
            (w // 2 - 80, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    # State banner at bottom
    state_color = STATE_COLORS.get(result.state, WHITE)
    banner_text = f"  {result.state.name}  "
    cv2.rectangle(frame, (0, h - 30), (w, h), state_color, -1)
    cv2.putText(
        frame,
        banner_text,
        (w // 2 - 40, h - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2,
    )


def main() -> None:
    """Run the webcam demo."""
    print("PilotGuard Webcam Demo")
    print("=" * 40)
    print("Controls: q=quit, c=recalibrate, space=pause")
    print()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam. Check camera connection.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    pipeline = CVPipeline(
        calibration_duration=10.0,  # 10s calibration for demo (faster than production 30s)
        frame_width=640,
        frame_height=480,
    )
    pipeline.start_session()

    fps_counter = 0
    fps_time = time.monotonic()
    fps_display = 0.0
    paused = False

    print("Starting... Look at the camera with eyes open for calibration.")

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("ERROR: Failed to read frame from webcam.")
                    break

                result = pipeline.process_frame(frame)

                # FPS calculation
                fps_counter += 1
                elapsed = time.monotonic() - fps_time
                if elapsed >= 1.0:
                    fps_display = fps_counter / elapsed
                    fps_counter = 0
                    fps_time = time.monotonic()

                # Draw landmarks and overlays
                if result.face_detected and result.landmarks_68:
                    h, w = frame.shape[:2]
                    draw_landmarks(frame, result.landmarks_68.points, w, h)

                    eye_color = RED if result.is_eyes_closed else GREEN
                    draw_eye_contour(frame, result.landmarks_68.right_eye, w, h, eye_color)
                    draw_eye_contour(frame, result.landmarks_68.left_eye, w, h, eye_color)

                draw_hud(frame, result, fps_display)

                cv2.imshow("PilotGuard Demo", frame)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):  # q or ESC
                break
            elif key == ord("c"):
                print("Restarting calibration...")
                pipeline.start_session()
            elif key == ord(" "):
                paused = not paused
                print("PAUSED" if paused else "RESUMED")

    finally:
        pipeline.close()
        cap.release()
        cv2.destroyAllWindows()
        print("\nDemo ended.")


if __name__ == "__main__":
    main()
