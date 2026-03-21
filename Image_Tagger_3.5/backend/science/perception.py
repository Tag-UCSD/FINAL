"""
High-Level AI Perception Logic.
Handles VLM prompts and Semantic Analysis.
"""
import os
import base64
from io import BytesIO
from PIL import Image
from backend.science.core import AnalysisFrame

class PerceptionProcessor:
    
    def __init__(self):
        # Check for keys but don't crash if missing (allow partial functionality)
        self.has_openai = "OPENAI_API_KEY" in os.environ
        self.has_anthropic = "ANTHROPIC_API_KEY" in os.environ

    async def analyze_aesthetics(self, frame: AnalysisFrame):
        """
        Uses VLM to extract abstract qualities (Modernity, Coziness).
        """
        if not self.has_openai:
            # Fallback for dev/test without cost
            frame.add_attribute('style.modernity', 0.5, confidence=0.0)
            return

        # Convert to base64
        pil_img = Image.fromarray(frame.original_image)
        buff = BytesIO()
        pil_img.save(buff, format="JPEG")
        b64_img = base64.b64encode(buff.getvalue()).decode('utf-8')
        
        # Mock VLM Call - In production, insert actual OpenAI/Claude call here
        # We stub the network call to keep this file runnable immediately
        # result = await call_gpt4v(b64_img, "Rate modernity 0-1")
        
        # Simulated result
        frame.add_attribute('style.modernity', 0.85, confidence=0.9)
        frame.add_attribute('psych.coziness', 0.42, confidence=0.8)