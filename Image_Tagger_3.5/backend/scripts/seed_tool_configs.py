import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy.orm import Session
from backend.database.core import SessionLocal, engine
from backend.models.config import ToolConfig, Base

def seed():
    # Ensure tables exist
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    
    print("🌱 Seeding Tool Configurations")
    
    tools = [
        # Segmentation Models
        {"name": "sam_vit_b", "type": "segmentation", "provider": "local", "cost_per_image": 0.0001, "settings": {"speed": "fast"}},
        {"name": "sam_vit_l", "type": "segmentation", "provider": "local", "cost_per_image": 0.0003, "settings": {"speed": "medium"}},
        {"name": "sam_vit_h", "type": "segmentation", "provider": "local", "cost_per_image": 0.0008, "settings": {"speed": "slow"}},
        
        # VLM Labeling Models
        {"name": "gemini-1.5-flash", "type": "labeling", "provider": "google", "cost_per_1k_tokens": 0.0001, "settings": {"context": "1m"}},
        {"name": "gpt-4-vision", "type": "labeling", "provider": "openai", "cost_per_1k_tokens": 0.01, "settings": {"context": "128k"}},
        {"name": "claude-3-sonnet", "type": "labeling", "provider": "anthropic", "cost_per_1k_tokens": 0.003, "settings": {"context": "200k"}},
        {"name": "deepseek-vl", "type": "labeling", "provider": "local", "cost_per_1k_tokens": 0.0, "settings": {"quantization": "4bit"}},
    ]

    for tool in tools:
        exists = db.query(ToolConfig).filter_by(name=tool["name"]).first()
        if not exists:
            # Store tool type in settings since model doesn't have tool_type field
            settings = tool["settings"].copy()
            settings["type"] = tool["type"]
            t = ToolConfig(
                name=tool["name"],
                provider=tool["provider"],
                cost_per_image=tool.get("cost_per_image", 0.0),
                cost_per_1k_tokens=tool.get("cost_per_1k_tokens", 0.0),
                settings=settings,
                is_enabled=True
            )
            db.add(t)
            print(f"  + Added {tool['name']}")
        else:
            print(f"  . Skipped {tool['name']} (Exists)")
            
    db.commit()
    db.close()
    print("✅ Seeding Complete.")

if __name__ == "__main__":
    seed()