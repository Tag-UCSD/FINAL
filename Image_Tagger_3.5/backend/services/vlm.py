"""
Minimal VLM engine abstraction for the science pipeline.

Provides get_vlm_engine() which returns a configured VLM engine
(Gemini, OpenAI, Anthropic) or a StubEngine fallback.
"""
from __future__ import annotations

import os
import logging
from typing import Optional

logger = logging.getLogger("v3.services.vlm")


class StubEngine:
    """Fallback engine when no VLM API key is configured."""

    def analyze_image(self, image_data, prompt: str, **kwargs) -> dict:
        return {"stub": True, "error": "No VLM configured"}

    def __str__(self):
        return "StubEngine"


def get_vlm_engine(provider_override: Optional[str] = None):
    """Return a VLM engine based on available API keys."""
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if gemini_key:
        try:
            from google import genai
            client = genai.Client(api_key=gemini_key)

            class GeminiEngine:
                def analyze_image(self, image_data, prompt: str, **kwargs) -> dict:
                    import base64
                    from io import BytesIO
                    from PIL import Image as PILImage
                    import numpy as np

                    if isinstance(image_data, np.ndarray):
                        img = PILImage.fromarray(image_data)
                        buf = BytesIO()
                        img.save(buf, format="JPEG")
                        b64 = base64.b64encode(buf.getvalue()).decode()
                    else:
                        b64 = base64.b64encode(image_data).decode()

                    try:
                        response = client.models.generate_content(
                            model="gemini-2.0-flash",
                            contents=[
                                {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
                                prompt,
                            ],
                        )
                        return {"text": response.text, "stub": False}
                    except Exception as e:
                        return {"stub": True, "error": str(e)}

                def __str__(self):
                    return "Gemini Flash"

            return GeminiEngine()
        except Exception as e:
            logger.warning(f"Gemini init failed: {e}")

    return StubEngine()


def get_cognitive_prompt(base_prompt: str) -> str:
    """Return the prompt for cognitive analysis (passthrough)."""
    return base_prompt


def describe_vlm_configuration() -> dict:
    """Return a summary of the current VLM configuration."""
    engine = get_vlm_engine()
    return {
        "engine": str(engine),
        "is_stub": isinstance(engine, StubEngine),
    }
