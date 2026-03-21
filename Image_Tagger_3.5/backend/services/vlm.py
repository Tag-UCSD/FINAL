
"""Unified Visual Language Model (VLM) service.

This module abstracts different VLM providers (OpenAI, Anthropic, Gemini)
behind a single interface so the science pipeline can call a single
`get_vlm_engine()` entry point.

Design goals:
- Keep the pipeline code simple and synchronous.
- Prefer environment variables for API keys.
- Allow a lightweight runtime preference via a small config file.
- Fall back to a Stub engine that returns neutral placeholders when
  no keys are configured (useful for classrooms and tests).
"""

from __future__ import annotations

import base64
import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Optional SDK imports. These may not be installed in all environments;
# the Docker backend image installs them explicitly.
try:  # OpenAI >= 1.x
    import openai  # type: ignore
except Exception:  # pragma: no cover - optional
    openai = None  # type: ignore

try:  # Anthropic
    import anthropic  # type: ignore
except Exception:  # pragma: no cover - optional
    anthropic = None  # type: ignore

try:  # Gemini / Google Generative AI
    import google.generativeai as genai  # type: ignore
except Exception:  # pragma: no cover - optional
    genai = None  # type: ignore

# Where we store a tiny bit of runtime configuration.
_CONFIG_PATH = Path(__file__).resolve().parents[1] / "data" / "vlm_config.json"


def _safe_json_loads(raw: str) -> Dict[str, Any]:
    """Parse JSON from a VLM response with light, conservative repair.

    This helper is more forgiving than a single `json.loads` call, which can
    fail when a model wraps JSON in extra commentary or Markdown fences.

    Strategy:
    - Strip surrounding whitespace.
    - If code fences are present, extract the fenced block.
    - Try `json.loads` on the cleaned string.
    - On failure, look for the first '{' and last '}' and try that span.
    - If parsing still fails, re-raise the JSONDecodeError so callers see
      a clear failure rather than a silent mis-parse.

    This is intentionally conservative and does *not* attempt arbitrary
    "JSON repair"; it only handles the most common wrapping patterns.
    """
    cleaned = raw.strip()
    # Handle ```json or generic ``` fences.
    if "```json" in cleaned:
        try:
            cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0]
        except Exception:
            cleaned = cleaned.replace("```json", "").replace("```", "")
    elif "```" in cleaned:
        try:
            cleaned = cleaned.split("```", 1)[1].split("```", 1)[0]
        except Exception:
            cleaned = cleaned.replace("```", "")

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = cleaned[start : end + 1]
            return json.loads(snippet)
        # Re-raise original error if we cannot find a plausible JSON snippet.
        raise


class VLMEngine(ABC):
    """Abstract base class for a Visual Language Model provider."""

    @abstractmethod
    def analyze_image(self, image_bytes: bytes, prompt: str) -> Dict[str, Any]:
        """Return a JSON-serializable dict for the given image + prompt."""
        raise NotImplementedError

    @staticmethod
    def _encode_image(image_bytes: bytes) -> str:
        return base64.b64encode(image_bytes).decode("utf-8")


class StubEngine(VLMEngine):
    """Fallback engine when no API keys are configured.

    This is deliberately boring: it returns a small JSON stub so downstream
    code has the right shape but no one mistakes it for real analysis.
    """

    def analyze_image(self, image_bytes: bytes, prompt: str) -> Dict[str, Any]:
        logger.warning(
            "VLM StubEngine invoked (no real VLM provider configured). "
            "Returning placeholder values."
        )
        return {
            "stub": True,
            "note": (
                "This is dummy data from StubEngine. "
                "Configure GEMINI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY "
                "to enable real cognitive analysis."
            ),
        }


class OpenAIEngine(VLMEngine):
    """Adapter for OpenAI GPT-4o / GPT-4o-mini vision."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        if openai is None:
            raise RuntimeError("openai package is not installed in this environment.")
        # OpenAI 1.x client
        self._client = openai.OpenAI(api_key=api_key)  # type: ignore[attr-defined]
        self._model = model

    def analyze_image(self, image_bytes: bytes, prompt: str) -> Dict[str, Any]:
        b64_image = self._encode_image(image_bytes)
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt + " Return ONLY valid JSON.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64_image}"
                                },
                            },
                        ],
                    }
                ],
                response_format={"type": "json_object"},
                max_tokens=500,
            )
            raw_content = response.choices[0].message.content  # type: ignore[index]
            return _safe_json_loads(raw_content)
        except Exception as exc:  # pragma: no cover - network
            logger.error("OpenAI VLM error: %s", exc)
            raise


class AnthropicEngine(VLMEngine):
    """Adapter for Anthropic Claude 3.5 Sonnet."""

    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20240620"):
        if anthropic is None:
            raise RuntimeError(
                "anthropic package is not installed in this environment."
            )
        self._client = anthropic.Anthropic(api_key=api_key)  # type: ignore[attr-defined]
        self._model = model

    def analyze_image(self, image_bytes: bytes, prompt: str) -> Dict[str, Any]:
        b64_image = self._encode_image(image_bytes)
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": b64_image,
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt
                                + " Return strictly JSON (machine readable).",
                            },
                        ],
                    }
                ],
            )
            # Claude does not yet have strict JSON mode; parse text content.
            raw = response.content[0].text  # type: ignore[index]
            if "```json" in raw:
                raw = raw.split("```json", 1)[1].split("```", 1)[0]
            return _safe_json_loads(raw)
        except Exception as exc:  # pragma: no cover - network
            logger.error("Anthropic VLM error: %s", exc)
            raise


class GeminiEngine(VLMEngine):
    """Adapter for Google Gemini (e.g. 1.5 Flash / Pro) vision models."""

    def __init__(self, api_key: str, model: str = "gemini-3.0-flash"):
        if genai is None:
            raise RuntimeError(
                "google.generativeai package is not installed in this environment."
            )
        # Configure client; library will often also look at GEMINI_API_KEY env,
        # but we pass explicitly for clarity.
        try:
            # Newer SDK style
            from google import genai as google_genai  # type: ignore

            self._client = google_genai.Client(api_key=api_key)
            self._model_name = model
            self._mode = "client"
        except Exception:  # pragma: no cover - fallback to older style
            genai.configure(api_key=api_key)  # type: ignore[call-arg]
            self._model = genai.GenerativeModel(model)  # type: ignore[attr-defined]
            self._mode = "legacy"

    def analyze_image(self, image_bytes: bytes, prompt: str) -> Dict[str, Any]:
        try:
            if getattr(self, "_mode", "legacy") == "client":
                # Newer client style
                from google.genai.types import Part  # type: ignore

                image_part = Part.from_bytes(
                    data=image_bytes, mime_type="image/jpeg"
                )
                response = self._client.models.generate_content(
                    model=self._model_name,
                    contents=[prompt, image_part],
                )
                raw = response.text  # type: ignore[attr-defined]
            else:
                # Older google.generativeai style
                response = self._model.generate_content(  # type: ignore[attr-defined]
                    [
                        {"mime_type": "image/jpeg", "data": image_bytes},
                        prompt,
                    ]
                )
                raw = response.text  # type: ignore[attr-defined]

            if "```json" in raw:
                raw = raw.split("```json", 1)[1].split("```", 1)[0]
            return _safe_json_loads(raw)
        except Exception as exc:  # pragma: no cover - network
            logger.error("Gemini VLM error: %s", exc)
            raise


def _detect_available_backends() -> Dict[str, bool]:
    """Return which providers appear to be available based on API keys."""
    # Gemini can use GEMINI_API_KEY or GOOGLE_API_KEY depending on setup.
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    return {
        "gemini": bool(gemini_key),
        "openai": bool(os.getenv("OPENAI_API_KEY")),
        "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
    }


def _load_config() -> Dict[str, Any]:
    if _CONFIG_PATH.exists():
        try:
            return json.loads(_CONFIG_PATH.read_text())
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to read VLM config file: %s", exc)
    return {}


def _save_config(cfg: Dict[str, Any]) -> None:
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


def _resolve_provider(config_provider: Optional[str] = None,
                      override: Optional[str] = None) -> str:
    """Decide which provider string to use.

    Preference order:
    1. Explicit override argument (e.g. from /vlm/test payload).
    2. Config file provider (set via Admin UI).
    3. VLM_PROVIDER environment variable.
    4. Auto-detect based on available keys, preferring Gemini > OpenAI > Anthropic.
    5. Fallback to "stub".
    """
    # Normalise inputs
    if override:
        override = override.lower()
    if config_provider:
        config_provider = config_provider.lower()

    if override and override != "auto":
        return override

    if config_provider and config_provider != "auto":
        return config_provider

    env_provider = os.getenv("VLM_PROVIDER")
    if env_provider and env_provider.lower() != "auto":
        return env_provider.lower()

    # Auto-detect
    avail = _detect_available_backends()
    if avail.get("gemini"):
        return "gemini"
    if avail.get("openai"):
        return "openai"
    if avail.get("anthropic"):
        return "anthropic"
    return "stub"


def get_vlm_engine(provider_override: Optional[str] = None) -> VLMEngine:
    """Factory for a VLMEngine instance.

    provider_override can be None (use config/env), "auto", or one of:
    "gemini", "openai", "anthropic", "stub".
    """
    cfg = _load_config()
    config_provider = cfg.get("provider")
    provider = _resolve_provider(config_provider, provider_override)

    avail = _detect_available_backends()

    if provider == "openai" and avail.get("openai"):
        key = os.getenv("OPENAI_API_KEY")
        return OpenAIEngine(key)  # type: ignore[arg-type]
    if provider == "anthropic" and avail.get("anthropic"):
        key = os.getenv("ANTHROPIC_API_KEY")
        return AnthropicEngine(key)  # type: ignore[arg-type]
    if provider == "gemini" and avail.get("gemini"):
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        return GeminiEngine(key)  # type: ignore[arg-type]

    # If we reach here, fall back to stub.
    logger.info(
        "VLM falling back to StubEngine (provider=%s, avail=%s)",
        provider,
        avail,
    )
    return StubEngine()


def describe_vlm_configuration() -> Dict[str, Any]:
    """Return a small JSON-ready summary for the Admin cockpit.

    In addition to the active provider and detected backends, this also exposes
    optional ergonomics settings used by the scientist-facing tools:
    - cognitive_prompt_override: optional override for the cognitive/affective prompt
    - max_batch_size: soft limit for recommended images per batch
    - cost_per_1k_images_usd: rough cost estimate per 1000 images
    """
    cfg = _load_config()
    provider = cfg.get("provider", "auto")
    avail = _detect_available_backends()
    engine = get_vlm_engine()
    return {
        "provider": provider,
        "engine": type(engine).__name__,
        "available_backends": avail,
        "cognitive_prompt_override": cfg.get("cognitive_prompt_override") or "",
        "max_batch_size": cfg.get("max_batch_size"),
        "cost_per_1k_images_usd": cfg.get("cost_per_1k_images_usd"),
    }


def update_vlm_config(
    provider: Optional[str] = None,
    cognitive_prompt_override: Optional[str] = None,
    max_batch_size: Optional[int] = None,
    cost_per_1k_images_usd: Optional[float] = None,
) -> Dict[str, Any]:
    """Update the VLM configuration and return the updated description.

    This is intentionally forgiving:
    - empty / whitespace-only prompt clears any existing override
    - non-positive batch size or cost values clear those fields
    """
    cfg = _load_config()

    if provider is not None:
        cfg["provider"] = (provider or "auto").lower()

    if cognitive_prompt_override is not None:
        text = cognitive_prompt_override.strip()
        if text:
            cfg["cognitive_prompt_override"] = text
        else:
            cfg.pop("cognitive_prompt_override", None)

    if max_batch_size is not None:
        try:
            value = int(max_batch_size)
        except (TypeError, ValueError):
            value = None
        if value and value > 0:
            cfg["max_batch_size"] = value
        else:
            cfg.pop("max_batch_size", None)

    if cost_per_1k_images_usd is not None:
        try:
            value = float(cost_per_1k_images_usd)
        except (TypeError, ValueError):
            value = None
        if value and value > 0:
            cfg["cost_per_1k_images_usd"] = value
        else:
            cfg.pop("cost_per_1k_images_usd", None)

    _save_config(cfg)
    return describe_vlm_configuration()




def get_cognitive_prompt(base_prompt: str) -> str:
    """Return the effective cognitive/affective prompt.

    If the admin has configured a cognitive_prompt_override, that text is used;
    otherwise the provided base_prompt is returned unchanged.
    """
    cfg = _load_config()
    override = cfg.get("cognitive_prompt_override")
    if isinstance(override, str) and override.strip():
        return override
    return base_prompt


def set_configured_provider(provider: str) -> Dict[str, Any]:
    """Persist only the chosen provider and return the updated description.

    This is kept for backwards compatibility; the Admin cockpit now prefers
    update_vlm_config so it can also configure prompt overrides and batch size.
    """
    return update_vlm_config(provider=provider or "auto")

