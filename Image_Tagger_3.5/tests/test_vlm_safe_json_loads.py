import json

import pytest

from backend.services import vlm


def test_safe_json_loads_plain_object():
    obj = {"a": 1, "b": "x"}
    raw = json.dumps(obj)
    parsed = vlm._safe_json_loads(raw)
    assert parsed == obj


def test_safe_json_loads_fenced_json_block():
    obj = {"foo": 123}
    raw = "```json\n" + json.dumps(obj) + "\n```"
    parsed = vlm._safe_json_loads(raw)
    assert parsed == obj


def test_safe_json_loads_generic_fence_block():
    obj = {"bar": 42}
    raw = "```" + json.dumps(obj) + "```"
    parsed = vlm._safe_json_loads(raw)
    assert parsed == obj


def test_safe_json_loads_with_leading_and_trailing_commentary():
    obj = {"baz": "ok"}
    inner = json.dumps(obj)
    raw = "model says:\n" + inner + "\nthanks"
    parsed = vlm._safe_json_loads(raw)
    assert parsed == obj


def test_safe_json_loads_raises_on_non_json():
    with pytest.raises(json.JSONDecodeError):
        vlm._safe_json_loads("this is not json at all")
