import cv2
import time
import sys
import argparse
import config as config
from detector import Detector


def open_stream(source, max_retries=5, retry_delay=2.0):
    """
    Opens a cv2.VideoCapture with retry logic.
    Critical for RTSP streams that drop and reconnect.
    """
    for attempt in range(1, max_retries + 1):
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            print(f"[pipeline] Stream opened: {source}")
            return cap
        print(f"[pipeline] Attempt {attempt}/{max_retries} failed. Retrying in {retry_delay}s …")
        time.sleep(retry_delay)
    raise RuntimeError(f"[pipeline] Could not open source after {max_retries} attempts: {source}")


def read_frame_safe(cap, source):
    """
    Read one frame; reconnect if the stream dropped.
    Returns (frame, cap) — cap may be a new object after reconnect.
    """
    ret, frame = cap.read()
    if ret:
        return frame, cap

    print("[pipeline] Frame read failed — reconnecting …")
    cap.release()
    time.sleep(1.0)
    return None, open_stream(source)



def draw_hud(frame, fps, counts, alert):
    h, w = frame.shape[:2]
    pad = 10

    # ── FPS ──
    if config.SHOW_FPS:
        fps_txt = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_txt, (pad, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 180), 2, cv2.LINE_AA)

    # ── Object counts panel ──
    if config.SHOW_COUNTS and counts:
        lines = [f"{cls}: {n}" for cls, n in sorted(counts.items())]
        panel_h = len(lines) * 24 + 12
        panel_w = max(len(l) for l in lines) * 12 + 20

        # semi-transparent dark chip
        overlay = frame.copy()
        cv2.rectangle(overlay, (pad, 40), (pad + panel_w, 40 + panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        for i, line in enumerate(lines):
            cv2.putText(frame, line, (pad + 8, 40 + 20 + i * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

    # Alert banner
    if alert:
        banner = "INTRUSION DETECTED"
        (bw, bh), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        bx = (w - bw) // 2
        by = h - 50

        overlay = frame.copy()
        cv2.rectangle(overlay, (bx - 14, by - bh - 8), (bx + bw + 14, by + 10), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.putText(frame, banner, (bx, by),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    return frame


#  Main loop


def run():
    detector = Detector()
    cap = open_stream(config.SOURCE)

    fps_timer = time.time()
    fps_count = 0
    fps_display = 0.0

    print("[pipeline] Running. Press Q in the window (or Ctrl-C) to stop.")

    while True:
        frame, cap = read_frame_safe(cap, config.SOURCE)
        if frame is None:
            continue

        # Inference
        annotated, counts, alert = detector.run(frame)

        # FPS computation
        fps_count += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            fps_display = fps_count / elapsed
            fps_count = 0
            fps_timer = time.time()

        #HUD
        output = draw_hud(annotated, fps_display, counts, alert)

        # Show
        cv2.imshow(config.WINDOW_NAME, output)

        # Terminal log (every second)
        if fps_count == 1:          # fires once per FPS cycle
            status = "ALERT" if alert else "clear"
            counts_str = ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "none"
            print(f"[FPS {fps_display:5.1f}]  objects: {counts_str}  |  zone: {status}")

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[pipeline] Stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default=None)
    args = parser.parse_args()

    if args.source:
        config.SOURCE = args.source
    try:
        run()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print("\n[pipeline] Interrupted.")
    except RuntimeError as e:
        print(e)
        sys.exit(1)
