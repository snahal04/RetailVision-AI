"""Manual YOLO tracking script — not run by pytest."""

from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.track(
    source="videos/Entry_CAM.mp4",
    tracker="bytetrack.yaml",
    persist=True,
    classes=[0],
    show=True,
    device=0,
)
