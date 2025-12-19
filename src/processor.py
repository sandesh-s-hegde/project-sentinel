import cv2
import numpy as np
from ultralytics import YOLO
import math

class SentinelProcessor:
    def __init__(self, model_size='n'):
        self.model = YOLO(f'yolov8{model_size}.pt')
        self.prev_positions = {}

    def calculate_entropy(self, vectors, bins=12):
        """
        Calculates Shannon Entropy of flow direction.
        High Entropy = Chaotic/Volatile flow.
        """
        if not vectors:
            return 0.0
        
        # Convert vectors to angles (0-360)
        angles = [math.degrees(math.atan2(dy, dx)) % 360 for dx, dy in vectors]
        
        # Histogram binning
        hist, _ = np.histogram(angles, bins=bins, range=(0, 360))
        
        # Normalize to probability distribution
        total = sum(hist)
        if total == 0: return 0.0
        probs = hist / total
        
        # Shannon Entropy Formula: -Sum(p * log2(p))
        return -sum(p * math.log2(p) for p in probs if p > 0)

    def process_video_feed(self, video_path, status_callback=None):
        """
        Process entire video and return volatility metrics.
        """
        cap = cv2.VideoCapture(video_path)
        entropy_metrics = []
        volatility_proxies = [] # Sigma values
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame_count += 1
            if status_callback: status_callback(frame_count)

            # Tracking Logic
            results = self.model.track(frame, persist=True, verbose=False)
            current_vectors = []
            current_positions = {}
            
            if results[0].boxes.id is not None:
                ids = results[0].boxes.id.int().cpu().tolist()
                boxes = results[0].boxes.xywh.cpu()
                
                for box, track_id in zip(boxes, ids):
                    x, y, w, h = box
                    center = (float(x), float(y))
                    current_positions[track_id] = center
                    
                    if track_id in self.prev_positions:
                        px, py = self.prev_positions[track_id]
                        dx, dy = center[0] - px, center[1] - py
                        if abs(dx) > 0.5 or abs(dy) > 0.5:
                            current_vectors.append((dx, dy))

            self.prev_positions = current_positions
            
            # Calculate Metrics
            H = self.calculate_entropy(current_vectors)
            entropy_metrics.append(H)
            
            # Normalize Entropy to Sigma (Volatility Proxy)
            # Map 0->3.58 Entropy to 0.1->0.6 Sigma
            max_entropy = math.log2(12)
            sigma = 0.1 + (H / max_entropy) * 0.5
            volatility_proxies.append(sigma)

        cap.release()
        return entropy_metrics, volatility_proxies