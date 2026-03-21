"""
Import images from a JSON file into the database.

Usage:
    python backend/scripts/import_images_from_json.py <json_file>

Expected JSON format:
{
    "images": [
        {
            "filename": "image1.jpg",
            "url": "https://example.com/image1.jpg",  // or local path
            "tags": ["Modern", "Kitchen"],
            "meta_data": {
                "width": 1920,
                "height": 1080,
                "source": "google_images"
            }
        },
        ...
    ]
}

Alternative flat format (array of images):
[
    {
        "filename": "image1.jpg",
        "url": "https://example.com/image1.jpg",
        ...
    }
]
"""
import sys
import os
import json
import argparse
import uuid
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.database.core import SessionLocal, engine, Base
from backend.models.assets import Image


def import_images(json_path: str, batch_id: str = None) -> dict:
    """
    Import images from a JSON file.
    
    Returns a summary dict with counts.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Handle both formats: {"images": [...]} or [...]
    if isinstance(data, dict) and "images" in data:
        images_data = data["images"]
    elif isinstance(data, list):
        images_data = data
    else:
        raise ValueError("JSON must be either an array of images or an object with 'images' key")
    
    if not batch_id:
        batch_id = f"import_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    # Ensure tables exist
    Base.metadata.create_all(bind=engine)
    
    db = SessionLocal()
    created = 0
    skipped = 0
    errors = []
    
    try:
        for i, img_data in enumerate(images_data):
            try:
                # Extract fields
                filename = img_data.get("filename") or img_data.get("name") or f"image_{i}.jpg"
                url = img_data.get("url") or img_data.get("path") or img_data.get("storage_path")
                tags = img_data.get("tags", [])
                meta = img_data.get("meta_data") or img_data.get("metadata") or {}
                
                # Build storage_path from URL or provided path
                if url:
                    storage_path = url
                else:
                    storage_path = f"/static/images/{filename}"
                
                # Add tags to meta_data if not already there
                if tags and "tags" not in meta:
                    meta["tags"] = tags
                
                # Check if already exists (by filename)
                exists = db.query(Image).filter_by(filename=filename).first()
                if exists:
                    skipped += 1
                    continue
                
                img = Image(
                    filename=filename,
                    storage_path=storage_path,
                    meta_data=meta,
                    upload_batch_id=batch_id
                )
                db.add(img)
                created += 1
                
            except Exception as e:
                errors.append(f"Row {i}: {str(e)}")
        
        db.commit()
        
    finally:
        db.close()
    
    return {
        "batch_id": batch_id,
        "created": created,
        "skipped": skipped,
        "errors": errors,
        "total_in_file": len(images_data)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Import images from a JSON file into the database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example JSON format:
{
    "images": [
        {
            "filename": "kitchen_modern.jpg",
            "url": "https://example.com/images/kitchen_modern.jpg",
            "tags": ["Modern", "Kitchen", "High-Res"],
            "meta_data": {"width": 1920, "height": 1280}
        }
    ]
}
        """
    )
    parser.add_argument("json_file", help="Path to JSON file containing image data")
    parser.add_argument("--batch-id", help="Custom batch ID (auto-generated if not provided)")
    
    args = parser.parse_args()
    
    print(f"📥 Importing images from: {args.json_file}")
    
    try:
        result = import_images(args.json_file, args.batch_id)
        
        print(f"\n✅ Import Complete")
        print(f"   Batch ID: {result['batch_id']}")
        print(f"   Created:  {result['created']}")
        print(f"   Skipped:  {result['skipped']} (already exist)")
        print(f"   Total:    {result['total_in_file']}")
        
        if result['errors']:
            print(f"\n⚠️  Errors ({len(result['errors'])}):")
            for err in result['errors'][:10]:
                print(f"   - {err}")
            if len(result['errors']) > 10:
                print(f"   ... and {len(result['errors']) - 10} more")
                
    except Exception as e:
        print(f"\n❌ Import failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

