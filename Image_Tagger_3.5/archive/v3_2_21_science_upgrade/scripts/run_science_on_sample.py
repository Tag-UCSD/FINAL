#!/usr/bin/env python3
"""Run the science pipeline on a single sample image.

This is a convenience script to verify end-to-end wiring:

- Database connectivity via SQLAlchemy
- Image loading via the data_store service
- Science analyzers (complexity, texture, fractals, materials, color, context)
- Persistence of attributes into Validation rows
"""

import argparse
import asyncio
import logging

from backend.database.core import SessionLocal
from backend.science.pipeline import SciencePipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def _run(image_id: int) -> None:
    db = SessionLocal()
    try:
        pipeline = SciencePipeline(db)
        ok = await pipeline.process_image(image_id=image_id)
        if not ok:
            logger.error("Science pipeline failed for image_id=%s", image_id)
        else:
            logger.info("Science pipeline completed for image_id=%s", image_id)
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-id", type=int, required=True, help="ID of the Image row to process.")
    args = parser.parse_args()
    asyncio.run(_run(args.image_id))


if __name__ == "__main__":
    main()