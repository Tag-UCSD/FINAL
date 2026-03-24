"""
Minimal cost logging stub for science pipeline VLM calls.
"""
import logging

logger = logging.getLogger("v3.services.costs")


def log_vlm_usage(tool_name: str = "", provider: str = "", model_name: str = "",
                  cost_usd: float = 0.0, **kwargs):
    """Log VLM usage (no-op in minimal build, just logs to console)."""
    logger.info(f"VLM usage: {tool_name} via {provider}/{model_name} cost=${cost_usd:.4f}")
