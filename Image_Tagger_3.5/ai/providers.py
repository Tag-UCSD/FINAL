
# Minimal provider shim. Extend to call your preferred LLM vendor securely.
from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class ProviderConfig:
    name: str = "none"
    api_key: str = ""

    @staticmethod
    def from_env(name: str) -> "ProviderConfig":
        key = ""
        if name.lower() == "openai":
            key = ""  # read from env if you wire it
        return ProviderConfig(name=name, api_key=key)

class DummyClient:
    def __init__(self, cfg: ProviderConfig):
        self.cfg = cfg

    def propose_plan(self, redacted_log: str, policy: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "summary": f"Dummy plan via {self.cfg.name} (no external calls)",
            "steps": [{"action":"advice","text":"Wire a real provider in providers.py and set provider flag"}]
        }

def make_llm_client(cfg: ProviderConfig):
    return DummyClient(cfg)
