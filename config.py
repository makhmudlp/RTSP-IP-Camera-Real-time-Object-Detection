
#  config.py  —  single place to change everything


# Video source
#   0          → default webcam
#   1, 2, …    → other USB cameras
#   "rtsp://user:pass@192.168.1.x:554/stream" → real IP camera
#   "video.mp4"  → local file (great for demos)
SOURCE = 0

# YOLOv8 model  (auto-downloads on first run)
#   yolov8n  fastest / smallest
#   yolov8s  good balance
#   yolov8m  more accurate
MODEL = "yolov8n.pt"

# Inference settings
CONFIDENCE = 0.40          # minimum detection confidence
IOU_THRESHOLD = 0.45       # NMS IoU threshold
IMGSZ = 640                # inference resolution

# Alert zone  — list of (x, y) points as FRACTIONS of frame size (0.0–1.0)
# Default: centre rectangle covering ~25% of the frame
# Set to [] to disable zone alerting
ALERT_ZONE = [
    (0.35, 0.35),
    (0.65, 0.35),
    (0.65, 0.65),
    (0.35, 0.65),
]

# Which classes trigger the zone alert (COCO names). Empty list = all classes.
ALERT_CLASSES = ["person"]

# Display
SHOW_FPS        = True
SHOW_COUNTS     = True
SHOW_ZONE       = True
WINDOW_NAME     = "CV Pipeline — press Q to quit"
