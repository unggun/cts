"""Tests for parsing Claude review responses."""
from types import SimpleNamespace

from src.learning.claude_review import DEFAULT_MODEL, extract_review


def _msg(blocks, stop_reason="end_turn"):
    return SimpleNamespace(content=blocks, stop_reason=stop_reason)


def _text(t):
    return SimpleNamespace(type="text", text=t)


def test_default_model_is_current():
    assert DEFAULT_MODEL == "claude-sonnet-5-5"


def test_extract_review_skips_leading_thinking_block():
    thinking = SimpleNamespace(type="thinking", thinking="")
    msg = _msg([thinking, _text('Here:\n```json\n{"confidence": "high"}\n```')])
    assert extract_review(msg) == {"confidence": "high"}


def test_extract_review_joins_multiple_text_blocks():
    msg = _msg([_text('{"confidence":'), _text(' "medium"}')])
    assert extract_review(msg) == {"confidence": "medium"}


def test_extract_review_refusal_is_error():
    msg = _msg([], stop_reason="refusal")
    assert "error" in extract_review(msg)


def test_extract_review_bad_json_returns_raw():
    msg = _msg([_text("no json here")])
    assert extract_review(msg) == {"raw_response": "no json here"}
