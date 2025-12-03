#!/usr/bin/env python3
"""
Simple local inference tester for Unsloth LoRA adapters produced by FineFoundry.

- Loads a 4-bit base model (default: unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit).
- Applies a LoRA adapter from a training output directory (default: outputs/local_run/adapter
  under a given data root matching the /data mount used in Docker training).
- Provides a basic REPL: type a prompt, get a model completion.

This script is intentionally standalone and not imported by the main app.
It requires the `unsloth` and `torch` packages to be installed in your environment.
"""

import argparse
import os
import sys
from typing import Optional


def _eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def _load_model(base_model: str, adapter_path: str):
    try:
        import torch  # type: ignore[import]
    except Exception as ex:  # pragma: no cover - import guard
        _eprint("ERROR: PyTorch is not installed in this environment.")
        _eprint("Install it with: uv add torch")
        _eprint(f"Details: {ex}")
        sys.exit(1)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # type: ignore[import]
    except Exception as ex:  # pragma: no cover - import guard
        _eprint("ERROR: transformers is not installed in this environment.")
        _eprint("Install it with: uv add transformers")
        _eprint(f"Details: {ex}")
        sys.exit(1)

    try:
        from peft import PeftModel  # type: ignore[import]
    except Exception as ex:  # pragma: no cover - import guard
        _eprint("ERROR: peft is not installed in this environment.")
        _eprint("Install it with: uv add peft")
        _eprint(f"Details: {ex}")
        sys.exit(1)

    if not os.path.isdir(adapter_path):
        _eprint(f"ERROR: Adapter path does not exist or is not a directory: {adapter_path}")
        _eprint("Make sure you pointed --data-root to the same host directory you used as /data for training.")
        sys.exit(1)

    _eprint(f"Loading base model with transformers: {base_model}")
    model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if torch.cuda.is_available():
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=model_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                quantization_config=bnb_config,
                device_map="auto",
            )
        except Exception as ex:
            _eprint(f"Warning: 4-bit loading failed, falling back to full precision: {ex}")
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=model_dtype,
                device_map="auto",
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=model_dtype,
        )

    _eprint(f"Applying LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)

    try:
        device = model.device
    except Exception:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

    _eprint(f"Model ready on device: {device}")
    return model, tokenizer


def _generate(model, tokenizer, prompt: str, max_new_tokens: int = 256, temperature: float = 0.7) -> str:
    import torch  # local import to avoid hard dependency at module import time

    text = prompt.rstrip("\n")
    if not text:
        return ""

    inputs = tokenizer([text], return_tensors="pt")
    try:
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    except Exception:
        pass

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            do_sample=True,
            top_p=0.9,
        )
    out = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return out


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Test local Unsloth LoRA adapter inference.")
    parser.add_argument(
        "--data-root",
        default=os.path.expanduser("~/Desktop/test_data"),
        help=(
            "Host directory that was mounted as /data during training. "
            "The script will look under DATA_ROOT/outputs/local_run/adapter by default."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/local_run",
        help="Relative output directory under data-root where training wrote artifacts.",
    )
    parser.add_argument(
        "--adapter-subdir",
        default="adapter",
        help="Subdirectory under output-dir containing the LoRA adapter.",
    )
    parser.add_argument(
        "--base-model",
        default="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        help="Base model name to load.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate per prompt.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )

    args = parser.parse_args(argv)

    data_root = os.path.abspath(os.path.expanduser(args.data_root))
    adapter_path = os.path.join(data_root, args.output_dir, args.adapter_subdir)

    _eprint(f"Data root: {data_root}")
    _eprint(f"Adapter path: {adapter_path}")

    model, tokenizer = _load_model(args.base_model, adapter_path)

    _eprint("\nEnter prompts to test the fine-tuned model. Press Ctrl+C or an empty line to exit.\n")
    try:
        while True:
            try:
                prompt = input("Prompt> ")
            except EOFError:
                break
            if not prompt.strip():
                break
            _eprint("Generating...\n")
            out = _generate(model, tokenizer, prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
            print("\n=== Completion ===")
            print(out)
            print("\n==================\n")
    except KeyboardInterrupt:
        _eprint("\nInterrupted by user.")

    return 0


if __name__ == "__main__":  # pragma: no cover - script entry
    raise SystemExit(main())