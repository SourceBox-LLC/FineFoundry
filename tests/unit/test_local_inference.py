"""Unit tests for helpers.local_inference.generate_text.

These tests mock out heavy model/tokenizer loading and only verify:
- Prompt / chat_history shaping into messages
- Chat template vs. fallback formatting behavior
- Forwarding of max_new_tokens, temperature, repetition_penalty
- Early return on a truly empty prompt (""), matching the function contract
"""

from __future__ import annotations

from typing import Any, Dict, List

import helpers.local_inference as li


class DummyModel:
    """Minimal dummy model that records generate() kwargs."""

    def __init__(self) -> None:
        self.device = "cpu"
        self.last_kwargs: Dict[str, Any] | None = None

    def generate(self, **kwargs: Any):  # type: ignore[override]
        self.last_kwargs = kwargs
        # Return a single "sequence" of token ids  [1, 2, 3, 4, 5]
        return [[1, 2, 3, 4, 5]]


class DummyInputIds:
    """Minimal object to mimic a tensor with shape attribute."""

    def __init__(self, length: int) -> None:
        self.shape = (1, length)


class DummyTokenizerWithTemplate:
    """Tokenizer that exposes apply_chat_template path."""

    def __init__(self, input_len: int = 3) -> None:
        self.chat_template = "dummy-template"
        self._input_len = input_len
        self.last_messages: List[Dict[str, str]] | None = None
        self.last_tokenize: bool | None = None
        self.last_add_generation_prompt: bool | None = None
        self.last_inputs: List[str] | None = None
        self.last_decoded_tokens: Any | None = None

    def apply_chat_template(self, messages, tokenize: bool, add_generation_prompt: bool):  # type: ignore[override]
        self.last_messages = list(messages)
        self.last_tokenize = tokenize
        self.last_add_generation_prompt = add_generation_prompt
        # Just embed role + content so tests can inspect
        first = messages[0]
        return f"TEMPLATE::{first['role']}::{first['content']}"

    def __call__(self, texts, return_tensors: str = "pt"):  # type: ignore[override]
        # texts is expected to be a list with a single formatted prompt string
        self.last_inputs = list(texts)
        return {"input_ids": DummyInputIds(self._input_len)}

    def decode(self, tokens, skip_special_tokens: bool = True):  # type: ignore[override]
        self.last_decoded_tokens = tokens
        return f"DECODED_{list(tokens)}"


class DummyTokenizerNoTemplate:
    """Tokenizer without chat template support (fallback path)."""

    def __init__(self, input_len: int = 2) -> None:
        self._input_len = input_len
        self.last_inputs: List[str] | None = None
        self.last_decoded_tokens: Any | None = None

    def __call__(self, texts, return_tensors: str = "pt"):  # type: ignore[override]
        self.last_inputs = list(texts)
        return {"input_ids": DummyInputIds(self._input_len)}

    def decode(self, tokens, skip_special_tokens: bool = True):  # type: ignore[override]
        self.last_decoded_tokens = tokens
        return f"FALLBACK_DECODED_{list(tokens)}"


def test_generate_text_uses_chat_template_with_prompt(monkeypatch):
    """When no chat_history is provided, prompt is wrapped into messages and chat template is used."""

    model = DummyModel()
    tok = DummyTokenizerWithTemplate(input_len=3)

    def fake_load_model(base_model: str, adapter_path: str):  # type: ignore[override]
        assert base_model == "base-model"
        assert adapter_path == "/adapter"
        return model, tok

    monkeypatch.setattr(li, "load_model", fake_load_model)

    out = li.generate_text(
        base_model="base-model",
        adapter_path="/adapter",
        prompt="Hello world",
        max_new_tokens=5,
        temperature=0.5,
        repetition_penalty=1.2,
        chat_history=None,
    )

    # Output comes from DummyTokenizer.decode
    assert out == "DECODED_[4, 5]"  # input_len=3 -> slice [3:] == [4,5]

    # Chat template was applied with expected messages structure
    assert tok.last_messages == [{"role": "user", "content": "Hello world"}]
    assert tok.last_tokenize is False
    assert tok.last_add_generation_prompt is True

    # Generation args forwarded correctly
    assert model.last_kwargs is not None
    assert model.last_kwargs["max_new_tokens"] == 5
    assert model.last_kwargs["temperature"] == 0.5
    assert model.last_kwargs["repetition_penalty"] == 1.2


def test_generate_text_uses_chat_history_and_ignores_prompt(monkeypatch):
    """When chat_history is provided, prompt is ignored and messages come from history."""

    model = DummyModel()
    tok = DummyTokenizerWithTemplate(input_len=2)

    def fake_load_model(base_model: str, adapter_path: str):  # type: ignore[override]
        return model, tok

    monkeypatch.setattr(li, "load_model", fake_load_model)

    history = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]

    out = li.generate_text(
        base_model="base-model",
        adapter_path="/adapter",
        prompt="THIS SHOULD BE IGNORED",
        chat_history=history,
    )

    # Output still comes from DummyTokenizer.decode
    assert out == "DECODED_[3, 4, 5]"  # input_len=2 -> slice [2:] == [3,4,5]

    # Messages passed to chat template match chat_history exactly
    assert tok.last_messages == history


def test_generate_text_fallback_format_without_chat_template(monkeypatch):
    """If tokenizer has no chat template, a simple 'User/Assistant' format is used."""

    model = DummyModel()
    tok = DummyTokenizerNoTemplate(input_len=2)

    def fake_load_model(base_model: str, adapter_path: str):  # type: ignore[override]
        return model, tok

    monkeypatch.setattr(li, "load_model", fake_load_model)

    out = li.generate_text(
        base_model="base-model",
        adapter_path="/adapter",
        prompt="Explain transformers",
    )

    # Output from fallback tokenizer
    assert out == "FALLBACK_DECODED_[3, 4, 5]"

    # Fallback format should look like "User: ...\nAssistant:"
    assert tok.last_inputs is not None
    assert tok.last_inputs[0].startswith("User: Explain transformers")
    assert tok.last_inputs[0].endswith("\nAssistant:")


def test_generate_text_early_return_on_empty_prompt(monkeypatch):
    """Truly empty prompt ("") with no chat history should short-circuit before generate()."""

    class ModelShouldNotBeCalled(DummyModel):
        def generate(self, **kwargs: Any):  # type: ignore[override]
            raise AssertionError("generate() should not be called for empty prompt")

    model = ModelShouldNotBeCalled()
    tok = DummyTokenizerWithTemplate(input_len=3)

    def fake_load_model(base_model: str, adapter_path: str):  # type: ignore[override]
        return model, tok

    monkeypatch.setattr(li, "load_model", fake_load_model)

    out = li.generate_text(
        base_model="base-model",
        adapter_path="/adapter",
        prompt="",  # truly empty
    )

    assert out == ""
    # No tokens should have been decoded and generate() should not have been invoked
    assert tok.last_decoded_tokens is None
    assert model.last_kwargs is None


def test_generate_text_defaults_for_optional_args(monkeypatch):
    """Ensure default values for optional args are forwarded correctly."""

    model = DummyModel()
    tok = DummyTokenizerWithTemplate(input_len=3)

    def fake_load_model(base_model: str, adapter_path: str):  # type: ignore[override]
        return model, tok

    monkeypatch.setattr(li, "load_model", fake_load_model)

    _ = li.generate_text(
        base_model="base-model",
        adapter_path="/adapter",
        prompt="Hello",
    )

    assert model.last_kwargs is not None
    # Defaults from implementation: max_new_tokens=256, temperature=0.7, repetition_penalty=1.15
    assert model.last_kwargs["max_new_tokens"] == 256
    assert model.last_kwargs["temperature"] == 0.7
    assert model.last_kwargs["repetition_penalty"] == 1.15
