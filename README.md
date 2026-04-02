# Real-Time Video Analytics Pipeline

A production-style computer vision pipeline for live video streams — webcam, IP camera, or RTSP feed. Built with YOLOv8 and OpenCV.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green?style=flat-square)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-red?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## Features

- **Real-time object detection** — YOLOv8 inference on every frame with bounding boxes, class labels, and confidence scores
- **Alert zone** — define a polygon region; triggers a visual alert when a target class enters it
- **Stream reconnection** — automatically recovers from dropped RTSP/IP camera connections
- **FPS counter** — live performance monitoring overlaid on the video
- **Per-class object counts** — real-time HUD showing how many of each class are detected
- **Flexible source** — works with webcam, video file, RTSP stream, or YouTube stream URL

---

## Demo

```
[FPS  28.4]  objects: person=3, car=1  |  zone: ALERT
[FPS  29.1]  objects: person=2         |  zone: clear
```

---

## Project Structure

```
rtsp_pipeline/
├── pipeline.py       # main loop: capture → inference → display
├── detector.py       # YOLOv8 wrapper + zone intrusion logic
├── config.py         # all settings in one place
└── requirements.txt
```

---

## Quickstart

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/RTSP-IP-Camera-Real-time-Object-Detection.git
cd RTSP-IP-Camera-Real-time-Object-Detection
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run**
```bash
# Default: opens webcam
python pipeline.py

# Custom source via argument
python pipeline.py --source "rtsp://admin:pass@192.168.1.64:554/stream"
python pipeline.py --source "video.mp4"
python pipeline.py --source "https://yt-dlp-stream-url"
```

Press **Q** or **Escape** to quit.

---

## Configuration

All settings are in `config.py` — no need to touch the source code.

| Parameter | Default | Description |
|---|---|---|
| `SOURCE` | `0` | Webcam index, RTSP URL, or video file path |
| `MODEL` | `yolov8n.pt` | YOLOv8 variant — `n` (fastest) → `x` (most accurate) |
| `CONFIDENCE` | `0.40` | Minimum detection confidence (0.0 – 1.0) |
| `IOU_THRESHOLD` | `0.45` | NMS overlap threshold for duplicate box removal |
| `IMGSZ` | `640` | Inference resolution |
| `ALERT_ZONE` | centre rect | Polygon as fractions of frame size (0.0 – 1.0) |
| `ALERT_CLASSES` | `["person"]` | Classes that trigger zone alert. Empty = all classes |

### Example: RTSP IP camera
```python
SOURCE = "rtsp://admin:password@192.168.1.64:554/Streaming/Channels/101"
```

### Example: custom alert zone (bottom-left corner)
```python
ALERT_ZONE = [
    (0.0, 0.7),
    (0.4, 0.7),
    (0.4, 1.0),
    (0.0, 1.0),
]
```

### Example: alert on any class
```python
ALERT_CLASSES = []   # empty = all classes trigger the alert
```

---

## Using a YouTube stream (for testing)

```bash
pip install yt-dlp

# Get stream URL
yt-dlp -f best -g "https://www.youtube.com/watch?v=VIDEO_ID"

# Pass it directly
python pipeline.py --source "https://..."
```

> Note: YouTube stream URLs expire after a few hours. Re-run `yt-dlp` to get a fresh one.

---

## Model Size vs Speed Tradeoff

| Model | Size | Speed (CPU) | Use case |
|---|---|---|---|
| `yolov8n.pt` | 6 MB | ~30 FPS | Real-time on CPU |
| `yolov8s.pt` | 22 MB | ~20 FPS | Balanced |
| `yolov8m.pt` | 52 MB | ~10 FPS | Higher accuracy, GPU recommended |
| `yolov8l.pt` | 87 MB | ~5 FPS | GPU only |

Models download automatically on first run.

---

## Key Engineering Decisions

**Why fractional zone coordinates?**
Zone stays correct regardless of resolution — same config works at 480p and 4K without any changes.

**Why reconnection logic?**
Real RTSP streams drop packets and lose connection on network hiccups. Production systems must recover automatically without human intervention.

**Why FPS averaged over 1 second?**
Per-frame FPS fluctuates wildly (28ms one frame, 35ms the next). Averaging over a full second gives a stable, readable number.

**Why `defaultdict(int)` for counts?**
Avoids manual key existence checks — automatically initializes any new class to 0 before incrementing.

---

## Requirements

- Python 3.8+
- OpenCV 4.8+
- Ultralytics (YOLOv8)
- NumPy

---

## License

MIT
