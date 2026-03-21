from backend.science.core import AnalysisFrame

class SocialDispositionAnalyzer:
    """
    Generates VLM Prompts specifically for Architectural Psychology.
    """
    
    PROMPT_TEMPLATE = """
    Analyze this architectural interior as an environmental psychologist.
    Assess the following social affordances on a scale of 0.0 to 1.0:
    
    1. Sociopetal (Encourages interaction): {sociopetal_score}
    2. Sociofugal (Discourages interaction): {sociofugal_score}
    3. Privacy (Visual/Acoustic isolation): {privacy_score}
    4. Hierarchy (Clear distinction of status): {hierarchy_score}
    
    Output strictly in JSON format: { "sociopetal": 0.X, "sociofugal": 0.X, "privacy": 0.X, "hierarchy": 0.X }
    """

    @staticmethod
    async def analyze(frame: AnalysisFrame, perception_engine):
        """
        Uses the shared PerceptionProcessor (VLM) to run this specific study.
        """
        # In a real system, 'perception_engine' is the VLM client wrapper
        # result = await perception_engine.ask(frame.original_image, PROMPT_TEMPLATE)
        
        # MOCK RESULT (until VLM is live)
        frame.add_attribute("social.sociopetal", 0.75, confidence=0.8)
        frame.add_attribute("social.privacy", 0.30, confidence=0.8)