import numpy as np
from backend.science.core import AnalysisFrame
import logging

# Lazy load ultralytics to keep startup fast if not used
YOLO_MODEL = None

logger = logging.getLogger("v3.science.objects")

class ObjectAnalyzer:
    """
    Wraps YOLOv8 for counting architectural elements (chairs, people, plants).
    Essential for 'Social Affordance' metrics.
    """
    
    @staticmethod
    def load_model():
        global YOLO_MODEL
        if YOLO_MODEL is None:
            from ultralytics import YOLO
            # Downloads 'yolov8n.pt' (nano) automatically on first run (~6MB)
            # Use 'yolov8x.pt' for production accuracy
            logger.info("Loading YOLOv8 model...")
            YOLO_MODEL = YOLO("yolov8n.pt")
            
    @staticmethod
    def analyze(frame: AnalysisFrame):
        ObjectAnalyzer.load_model()
        
        # Run Inference
        results = YOLO_MODEL(frame.original_image, verbose=False)
        
        # Count classes
        counts = {}
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = YOLO_MODEL.names[cls_id]
                counts[label] = counts.get(label, 0) + 1
                
        # Mapping to CNfA attributes
        # 1. Seating (Social)
        seating_count = counts.get('chair', 0) + counts.get('couch', 0) + counts.get('bench', 0)
        frame.add_attribute("affordance.seating_count", seating_count)
        
        # 2. Biophilia (Plants)
        plant_count = counts.get('potted plant', 0)
        frame.add_attribute("biophilia.plant_count", plant_count)
        
        # 3. Occupancy
        person_count = counts.get('person', 0)
        frame.add_attribute("social.occupancy", person_count)