from backend.science.core import AnalysisFrame

class CognitiveStateAnalyzer:
    """
    Analyzes features affecting cognitive load and emotional state.
    """

    PROMPT_TEMPLATE = """
    Evaluate this space for its impact on human cognitive state:
    
    1. Restoration (Kaplan's ART): Likelihood of attention restoration.
    2. Mystery: Promise of more information if one moves deeper.
    3. Legibility: Ease of mapping the space mentally.
    4. Coherence: Order and organization of elements.
    
    Output strictly in JSON format.
    """

    @staticmethod
    async def analyze(frame: AnalysisFrame, perception_engine):
        # MOCK RESULT
        frame.add_attribute("cognitive.restoration", 0.65, confidence=0.6)
        frame.add_attribute("cognitive.mystery", 0.82, confidence=0.7)
        frame.add_attribute("cognitive.legibility", 0.90, confidence=0.9)