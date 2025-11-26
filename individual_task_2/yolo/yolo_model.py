from ultralytics import YOLO

class YoloModel:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.model = self._load_model(model_path)
    
    
    def _load_model(self, path):
        model = YOLO(path)
        model.to(self.device)
        return model
    

    def predict(self, frame, conf_threshold=0.75, size=512):
        if frame is None:
            return []
        
        results = self.model(frame, imgsz=size, conf=conf_threshold, verbose=False)

        detections = []
        boxes = results[0].boxes

        if boxes is None:
            return detections
        
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy()

        for box, conf_threshold, cls_id in zip(xyxy, confs, classes):
            detection = {
                "box": box.astype(int),
                "conf": float(conf_threshold),
                "cls": int(cls_id)
            }
            detections.append(detection)
        
        return detections
