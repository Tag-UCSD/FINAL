import cv2
import numpy as np
from sqlalchemy.orm import Session
from backend.models.assets import Image
from backend.models.annotation import Validation
from backend.science.core import AnalysisFrame

# Import our new modules
from backend.science.math.fractals import FractalAnalyzer
from backend.science.math.complexity import ComplexityAnalyzer
from backend.science.math.color import ColorAnalyzer
from backend.science.vision.materials import MaterialAnalyzer
from backend.science.context.social import SocialDispositionAnalyzer
from backend.science.context.cognitive import CognitiveStateAnalyzer
from backend.science.perception import PerceptionProcessor

class SciencePipeline:
    def __init__(self, db: Session):
        self.db = db
        self.perception = PerceptionProcessor()

    async def process_image(self, image_id: int):
        # 1. Load Image
        img_record = self.db.query(Image).filter(Image.id == image_id).first()
        if not img_record: return False
        
        # In production, load from S3/Disk
        # Mocking a blank image for logic verification if file missing
        try:
            local_path = f"data_store/{img_record.storage_path}"
            pixels = cv2.imread(local_path)
            pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
        except:
            pixels = np.zeros((800, 600, 3), dtype=np.uint8)

        frame = AnalysisFrame(image_id=image_id, original_image=pixels)

        # 2. Run Math (Fast / CPU)
        # Calculates Entropy, Edge Density, and Organization Ratio
        ComplexityAnalyzer.analyze(frame)
        
        # Calculates Global Fractal Dimension (D)
        d_score = FractalAnalyzer.fractal_dimension(frame.original_image)
        frame.add_attribute("fractal.D.global", d_score)
        # Material & color analysis (ported from v2 heuristics)
        MaterialAnalyzer.analyze(frame)
        ColorAnalyzer.analyze(frame)


        # 3. Run Context (Slow / GPU / VLM)
        await SocialDispositionAnalyzer.analyze(frame, self.perception)
        await CognitiveStateAnalyzer.analyze(frame, self.perception)

        # 4. Save to Database
        self._save_results(image_id, frame.attributes)
        
        return True

    def _save_results(self, image_id, attributes):
        for key, value in attributes.items():
            val = Validation(
                user_id=1, # System User
                image_id=image_id,
                attribute_key=key,
                value=value,
                duration_ms=0
            )
            self.db.add(val)
        self.db.commit()