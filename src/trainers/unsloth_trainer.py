from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def _hf_login_if_needed() -> None:
    if os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN"):
        return
    token_file = os.getenv("HF_TOKEN_FILE")
    if token_file and Path(token_file).exists():
        try:
            from huggingface_hub import login

            login(Path(token_file).read_text().strip())
        except Exception as e:
            print(f"[warn] HF login failed: {e}", file=sys.stderr)


def detect_format(data: List[Dict[str, Any]]) -> str:
    if not data:
        raise ValueError("Dataset is empty.")
    s = data[0]
    if "messages" in s and isinstance(s["messages"], list):
        return "chatml"
    if "conversations" in s and isinstance(s["conversations"], list):
        return "sharegpt"
    if "instruction" in s and "output" in s:
        return "instruction"
    if "input" in s and "output" in s:
        return "input_output"
    raise ValueError("Unknown dataset format")


def format_example(ex: Dict[str, Any], fmt: str, tokenizer) -> str:
    if fmt == "chatml":
        msgs = ex.get("messages", [])
        try:
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        except Exception:
            parts = [f"### {m['role'].title()}:\n{m['content']}" for m in msgs if isinstance(m, dict)]
            return "\n\n".join(parts) + tokenizer.eos_token
    if fmt == "sharegpt":
        convs = ex.get("conversations", [])
        parts: List[str] = []
        for c in convs:
            if not isinstance(c, dict):
                continue
            r = "User" if c.get("from") in ("human", "user") else "Assistant"
            parts.append(f"### {r}:\n{c.get('value', '')}")
        return "\n\n".join(parts) + tokenizer.eos_token
    if fmt == "instruction":
        inst = ex.get("instruction", "")
        inp = ex.get("input", "")
        out = ex.get("output", "")
        if inp:
            return f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}{tokenizer.eos_token}"
        return f"### Instruction:\n{inst}\n\n### Response:\n{out}{tokenizer.eos_token}"
    return f"### Input:\n{ex['input']}\n\n### Output:\n{ex['output']}{tokenizer.eos_token}"


def load_data(args, tokenizer):
    from datasets import Dataset, load_dataset

    rows: List[Dict[str, Any]]
    if args.hf_dataset_id:
        ds = load_dataset(args.hf_dataset_id, split=args.hf_dataset_split)
        rows = list(ds)
    elif args.json_path:
        with open(args.json_path, "r", encoding="utf-8-sig") as f:
            rows = json.load(f)
    else:
        raise ValueError("Provide --hf_dataset_id or --json_path")

    if not rows:
        raise ValueError("Dataset is empty.")

    fmt = detect_format(rows)
    print(f"[info] Detected format: {fmt}, {len(rows)} examples")
    texts = [format_example(r, fmt, tokenizer) for r in rows]
    return Dataset.from_dict({"text": texts})


def find_latest_checkpoint(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    newest: Optional[Path] = None
    newest_mtime = -1.0
    for run_dir in root.glob("*"):
        if not run_dir.is_dir():
            continue
        for ckpt in run_dir.glob("checkpoint-*"):
            try:
                m = ckpt.stat().st_mtime
            except Exception:
                continue
            if m > newest_mtime:
                newest, newest_mtime = ckpt, m
    return newest


def build_out_dir(args) -> Path:
    default_root = Path(os.getenv("OUTPUT_ROOT", "training_outputs"))
    stamp = time.strftime("%Y%m%d-%H%M%S")
    if args.output_dir:
        out = Path(args.output_dir)
        if not out.is_absolute():
            out = default_root / out
    else:
        out = default_root / stamp
    out.mkdir(parents=True, exist_ok=True)
    return out


def choose_optimizer(choice: Optional[str]) -> str:
    c = (choice or "auto").lower()
    if c == "adamw_8bit":
        try:
            import bitsandbytes  # noqa: F401

            return "adamw_8bit"
        except Exception as e:
            raise RuntimeError("bitsandbytes not installed") from e
    if c in ("adamw_torch", "adamw_hf"):
        return c
    try:
        import bitsandbytes  # noqa: F401

        return "adamw_8bit"
    except Exception:
        return "adamw_torch"


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()

    ap.add_argument("--json_path", default=None)
    ap.add_argument("--hf_dataset_id", default=None)
    ap.add_argument("--hf_dataset_split", default="train")

    ap.add_argument("--base_model", default="unsloth/Phi-3-mini-4k-instruct-bnb-4bit")

    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bsz", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--max_steps", type=int, default=-1)

    ap.add_argument("--packing", action="store_true", default=False)
    ap.add_argument("--report_to", default="none")
    ap.add_argument("--optim", default="auto")

    ap.add_argument("--lr_scheduler", default="cosine", choices=["linear", "cosine", "constant"])
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--warmup_steps", type=int, default=None)
    ap.add_argument("--weight_decay", type=float, default=0.01)

    ap.add_argument("--eval_every_steps", type=int, default=200)
    ap.add_argument("--save_every_steps", type=int, default=200)
    ap.add_argument("--save_total_limit", type=int, default=2)

    ap.add_argument("--use_lora", action="store_true", default=False)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=None)
    ap.add_argument("--lora_dropout", type=float, default=0.0)
    ap.add_argument("--use_rslora", action="store_true")

    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--resume_from", default=None)
    ap.add_argument("--auto_resume", action="store_true")

    ap.add_argument("--push", action="store_true")
    ap.add_argument("--hf_repo_id", default=None)
    ap.add_argument("--hf_private", action="store_true", default=True)

    return ap


def main(argv: Optional[List[str]] = None) -> int:
    _hf_login_if_needed()

    ap = build_arg_parser()
    args = ap.parse_args(argv)

    import inspect

    import unsloth  # noqa: F401
    import torch
    from transformers import TrainingArguments
    from trl import SFTTrainer
    from unsloth import FastLanguageModel

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    model, tok = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_len,
        dtype=None,
        load_in_4bit=True,
    )

    if args.use_lora:
        lora_alpha = args.lora_alpha if args.lora_alpha else args.lora_r * 2
        print(
            f"[info] LoRA config: r={args.lora_r}, alpha={lora_alpha}, dropout={args.lora_dropout}, rslora={args.use_rslora}"
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=args.use_rslora,
        )

    dset = load_data(args, tok)
    n = len(dset)
    n_eval = max(1, int(0.02 * n))
    d_train = dset.select(range(max(1, n - n_eval)))
    d_eval = dset.select(range(max(1, n - n_eval), n))

    out_dir = build_out_dir(args)

    ta_kwargs: Dict[str, Any] = dict(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum,
        lr_scheduler_type=args.lr_scheduler,
        weight_decay=args.weight_decay,
        logging_steps=25,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim=choose_optimizer(args.optim),
        eval_steps=args.eval_every_steps,
        save_strategy="steps",
        save_steps=args.save_every_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        max_steps=args.max_steps,
        report_to=args.report_to,
        dataloader_pin_memory=False,
        dataloader_num_workers=2,
        seed=3407,
    )

    if args.warmup_steps is not None and args.warmup_steps >= 0:
        ta_kwargs["warmup_steps"] = args.warmup_steps
    else:
        ta_kwargs["warmup_ratio"] = args.warmup_ratio

    try:
        args_hf = TrainingArguments(eval_strategy="steps", **ta_kwargs)
    except TypeError:
        args_hf = TrainingArguments(evaluation_strategy="steps", **ta_kwargs)

    _sft_kwargs: Dict[str, Any] = dict(
        model=model,
        train_dataset=d_train,
        eval_dataset=d_eval,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        dataset_num_proc=2,
        packing=args.packing,
        args=args_hf,
    )
    try:
        sig = inspect.signature(SFTTrainer.__init__)
        if "tokenizer" in sig.parameters:
            _sft_kwargs["tokenizer"] = tok
        elif "processing_class" in sig.parameters:
            _sft_kwargs["processing_class"] = tok
    except Exception:
        pass

    trainer = SFTTrainer(**_sft_kwargs)

    # FineFoundry handles publishing; avoid TRL's auto model card generation which can fail
    # if a previous run left a root-owned / read-only README.md in the output directory.
    try:
        trainer.create_model_card = lambda *a, **k: None  # type: ignore[assignment]
        print("[info] Disabled TRL model card generation (README.md)")
    except Exception:
        pass

    resume_path: Optional[Path] = None
    if args.resume_from:
        rp = Path(args.resume_from)
        if rp.exists():
            resume_path = rp
    elif args.auto_resume:
        cand = find_latest_checkpoint(out_dir)
        if not cand:
            cand = find_latest_checkpoint(Path(os.getenv("OUTPUT_ROOT", "training_outputs")))
        if cand:
            resume_path = cand
            print(f"[info] auto-resuming from: {resume_path}")

    stats = trainer.train(resume_from_checkpoint=str(resume_path) if resume_path else None)

    (out_dir / "adapter").mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir / "adapter"), safe_serialization=True)
    tok.save_pretrained(str(out_dir / "adapter"))

    try:
        (out_dir / "metrics.json").write_text(json.dumps(getattr(stats, "metrics", {}), indent=2))
    except Exception:
        pass

    try:
        (out_dir / "DONE").write_text("ok")
    except Exception:
        pass

    if args.push:
        print(
            "[warn] --push is deprecated in the integrated trainer. FineFoundry publishes adapters in the Publish tab.",
            file=sys.stderr,
        )

    print(f"[done] saved to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
