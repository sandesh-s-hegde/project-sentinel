import cv2
import numpy as np
from ultralytics import YOLO

class SentinelProcessor:
    def __init__(self, model_size='n'):
        """
        Initialize YOLOv8 model.
        Args:
            model_size (str): 'n' for nano (fastest), 's' for small.
        """
        print(f"Loading YOLOv8{model_size} model...")
        self.model = YOLO(f'yolov8{model_size}.pt')
        self.prev_positions = {}

    def process_frame(self, frame):
        """
        Tracks objects in a single frame.
        Returns list of flow vectors [(dx, dy), ...].
        """
        # Persist=True is critical for ID tracking across frames
        results = self.model.track(frame, persist=True, verbose=False)
        current_vectors = []
        current_positions = {}
        
        # Check if any objects were detected
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            ids = results[0].boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, ids):
                x, y, w, h = box
                center = (float(x), float(y))
                current_positions[track_id] = center
                
                # Calculate vector if object was seen in previous frame
                if track_id in self.prev_positions:
                    prev_x, prev_y = self.prev_positions[track_id]
                    dx = center[0] - prev_x
                    dy = center[1] - prev_y
                    
                    # Filter static noise (small jitter)
                    if abs(dx) > 0.5 or abs(dy) > 0.5:
                        current_vectors.append((dx, dy))

        self.prev_positions = current_positions
        return current_vectors