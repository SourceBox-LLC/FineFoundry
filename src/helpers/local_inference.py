from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


class LocalInferenceError(Exception):
    pass


_MODEL_CACHE: Dict[Tuple[str, str], Tuple[Any, Any]] = {}


def _cache_key(base_model: str, adapter_path: str) -> Tuple[str, str]:
    return (str(base_model), os.path.abspath(adapter_path))


def load_model(base_model: str, adapter_path: str):
    if not os.path.isdir(adapter_path):
        raise LocalInferenceError(f"Adapter path not found: {adapter_path}")

    key = _cache_key(base_model, adapter_path)
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    if use_cuda:
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                quantization_config=bnb_config,
                device_map="auto",
            )
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=dtype,
                device_map="auto",
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
        )

    model = PeftModel.from_pretrained(model, adapter_path)

    if not hasattr(model, "device"):
        device = "cuda" if use_cuda else "cpu"
        model.to(device)

    _MODEL_CACHE[key] = (model, tokenizer)
    return model, tokenizer


def generate_text(
    base_model: str,
    adapter_path: str,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> str:
    import torch as _torch

    text = (prompt or "").rstrip("\n")
    if not text:
        return ""

    model, tokenizer = load_model(base_model, adapter_path)

    inputs = tokenizer([text], return_tensors="pt")
    try:
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    except Exception:
        pass

    with _torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            do_sample=True,
            top_p=0.9,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
