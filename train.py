from ultralytics import YOLO

model = YOLO("yolov8m.pt")

results = model.train(
    data="chess.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="chess_detector",
    patience=10,
    augment=True,
    project="runs",
)
