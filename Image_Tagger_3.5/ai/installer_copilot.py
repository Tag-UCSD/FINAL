
#!/usr/bin/env python3
"""Read installer logs, redact secrets, propose a remediation plan, and write ai_plan.json.
This reference copilot does not call an external LLM by default; it builds a rule-based plan.
Integrators may extend providers.py to enable live model calls behind policy gates.
"""
import argparse, json, os, pathlib, re, sys
from typing import Dict, Any
from providers import make_llm_client, ProviderConfig  # optional

DEFAULT_POLICY = {
  "allow_shell": False,
  "allow_fs_write": True,
  "allow_network": False,
  "allowed_commands": ["pip install -r requirements.txt", "pytest -q"],
}

def redact_tokens(text: str) -> str:
    patterns = [
        r'(?i)(api[_-]?key|token|secret|passwd|password)\s*[:=]\s*[^\s]+',
        r'sk-[A-Za-z0-9]{20,}',
        r'AKIA[0-9A-Z]{16}',
        r'(?i)authorization:\s*bearer\s+[A-Za-z0-9._-]+',
        r'(?i)(x-api-key|x-auth-token)\s*[:=]\s*[^\s]+',
        r'(?:\?|&)token=[^&\s]+'
    ]
    redacted = text
    for pat in patterns:
        redacted = re.sub(pat, '<REDACTED>', redacted)
    return redacted

def simple_rule_plan(log_text: str) -> Dict[str, Any]:
    plan = {"summary": "Auto-generated remediation plan", "steps": [], "notes": []}
    t = log_text.lower()
    if "module not found" in t or "no module named" in t:
        plan["steps"].append({"action":"shell","cmd":"python -m pip install -r requirements.txt"})
    if "permission denied" in t:
        plan["steps"].append({"action":"advice","text":"Check file permissions and mark scripts executable (chmod +x *.sh)"})
    if "address already in use" in t:
        plan["steps"].append({"action":"advice","text":"Free the port or set a different port via env"})
    if "failed building wheel" in t or "error: subprocess-exited-with-error" in t:
        plan["steps"].append({"action":"advice","text":"Upgrade pip/setuptools/wheel then retry"})
        plan["steps"].append({"action":"shell","cmd":"python -m pip install --upgrade pip setuptools wheel"})
    if not plan["steps"]:
        plan["notes"].append("No specific errors recognized; propose running tests to collect traces.")
        plan["steps"].append({"action":"shell","cmd":"pytest -q"})
    return plan

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logfile', required=True)
    ap.add_argument('--out', default='logs/ai_plan.json')
    ap.add_argument('--dry-run', default='1')  # keep non-exec by default
    ap.add_argument('--provider', default='none')  # 'none' uses rule-based plan
    ap.add_argument('--policy', default=None)     # path to policy json
    args = ap.parse_args()

    root = pathlib.Path('.').resolve()
    log_text = pathlib.Path(args.logfile).read_text(encoding='utf-8', errors='ignore')
    log_text = redact_tokens(log_text)

    policy = DEFAULT_POLICY
    if args.policy and pathlib.Path(args.policy).exists():
        try:
            policy = json.loads(pathlib.Path(args.policy).read_text(encoding='utf-8'))
        except Exception:
            pass

    if args.provider == 'none':
        plan = simple_rule_plan(log_text)
    else:
        # Optional LLM call (must implement in providers.py and secure with policy)
        cfg = ProviderConfig.from_env(args.provider)
        client = make_llm_client(cfg)
        plan = client.propose_plan(log_text, policy)

    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps({
        "policy": policy,
        "dry_run": (args.dry_run == '1'),
        "plan": plan
    }, indent=2), encoding='utf-8')
    print(f"[copilot] wrote plan -> {outp}")

if __name__ == '__main__':
    main()
