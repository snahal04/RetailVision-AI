from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.track(
    source="videos\CAM 1.mp4",
    tracker="bytetrack.yaml",
    persist=True,
    classes=[0],
    show=True,
    device=0
)