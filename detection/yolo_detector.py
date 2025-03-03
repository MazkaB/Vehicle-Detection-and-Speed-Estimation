import torch
from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
    
    def predict(self, frame_rgb):
        results = self.model.predict(source=frame_rgb, conf=0.8, iou=0.5, verbose=False)
        return results[0]
