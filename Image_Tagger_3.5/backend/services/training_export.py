from typing import List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import select

from backend.models.assets import Image
from backend.models.annotation import Validation


class TrainingExporter:
    """Export human validations into a training dataset.

    This is the v3.2 counterpart to the v2.6.x TrainingDataExporter.
    It emits a list of JSON-serializable dictionaries that can be written
    as JSON or JSONL for fine-tuning / active learning loops.
    """
    def __init__(self, db: Session):
        self.db = db

    def export_for_images(self, image_ids: List[int]) -> List[Dict[str, Any]]:
        if not image_ids:
            return []

        stmt = (
            select(Validation, Image)
            .join(Image, Image.id == Validation.image_id)
            .where(Validation.image_id.in_(image_ids))
        )
        rows = self.db.execute(stmt).all()
        examples: List[Dict[str, Any]] = []

        for validation, image in rows:
            examples.append(
                {
                    "image_id": validation.image_id,
                    "image_filename": image.filename,
                    "attribute_key": validation.attribute_key,
                    "value": float(validation.value),
                    "user_id": validation.user_id,
                    "region_id": validation.region_id,
                    "duration_ms": validation.duration_ms,
                    "created_at": validation.created_at,
                    "source": "human_validation",
                }
            )

        return examples