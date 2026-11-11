from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="data/processed/yolo/data.yaml",
    epochs=1,
    imgsz=320,
    batch=2,
    device="cpu",
    workers=0,
    fraction=0.01,
    cache=False,
    project="runs/detect",
    name="yolo_smoke_test"
)