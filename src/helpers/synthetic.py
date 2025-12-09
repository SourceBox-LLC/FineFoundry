"""Synthetic data generation helper for FineFoundry.

Adapted from unsloth-synth-test - uses the exact same approach that works.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Dict, List, Optional
from urllib.parse import urlparse

import flet as ft

from helpers.common import safe_update
from helpers.scrape_db import save_scrape_to_db, save_chatml_to_db
from helpers.ui import two_col_header, two_col_row, compute_two_col_flex


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".html", ".htm", ".txt"}

TYPE_SUFFIXES = {
    "qa": "_qa_pairs",
    "cot": "_cot",
    "summary": "_summary",
}


def is_url(path: str) -> bool:
    """Check if the path is a URL."""
    try:
        result = urlparse(path)
        return result.scheme in ("http", "https")
    except Exception:
        return False


def create_config_file(output_folder: str = "data") -> Path:
    """Create synthetic-data-kit config file - exact same as working version."""
    config_path = Path("synthetic_data_kit_config.yaml")
    config_content = f"""# Synthetic Data Kit Configuration
output_folder: {output_folder}
vllm:
  base_url: "http://localhost:8000/v1"
"""
    config_path.write_text(config_content)
    return config_path


async def ingest_source_async(
    source_path: str,
    config_path: Path,
    multimodal: bool = False,
    log_fn: Optional[Callable] = None,
) -> Optional[Path]:
    """Ingest any supported source and extract text (async)."""
    source_type = "URL" if is_url(source_path) else Path(source_path).suffix.upper()
    if log_fn:
        await log_fn(f"📄 Ingesting ({source_type}): {Path(source_path).name}")

    cmd = ["synthetic-data-kit", "-c", str(config_path), "ingest", source_path]
    if multimodal:
        cmd.append("--multimodal")

    # Run subprocess in thread to not block event loop
    result = await asyncio.to_thread(subprocess.run, cmd, capture_output=True, text=True)

    if result.returncode != 0:
        if log_fn:
            stderr = result.stderr.strip()
            # Provide helpful error messages
            if "not found" in stderr.lower() or "No such file" in stderr:
                await log_fn(f"  ❌ File not found: {source_path}")
            elif "permission" in stderr.lower():
                await log_fn(f"  ❌ Permission denied: {source_path}")
            elif "unsupported" in stderr.lower() or "format" in stderr.lower():
                await log_fn(f"  ❌ Unsupported file format: {source_type}")
            elif "connection" in stderr.lower() or "timeout" in stderr.lower():
                await log_fn("  ❌ Network error fetching URL. Check connection.")
            else:
                await log_fn(f"  ❌ Ingestion failed: {stderr[:150]}")
        return None

    output_dir = Path("data/output")

    if is_url(source_path):
        parsed = urlparse(source_path)
        base_name = parsed.netloc.replace(".", "_")
    else:
        base_name = Path(source_path).stem.replace(" ", "_").replace("-", "_")

    patterns = [f"*{base_name}*", "*.txt", "*.lance"]
    for pattern in patterns:
        files = list(output_dir.glob(pattern))
        if files:
            return files[0]

    return None


async def generate_content_async(
    chunk_file: str,
    config_path: Path,
    gen_type: str = "qa",
    num_pairs: int = 25,
    log_fn: Optional[Callable] = None,
) -> Optional[Path]:
    """Generate content from a chunk file (async)."""
    if log_fn:
        await log_fn(f"  🔄 Generating {gen_type} from: {Path(chunk_file).name}")

    cmd = [
        "synthetic-data-kit",
        "-c",
        str(config_path),
        "create",
        chunk_file,
        "--type",
        gen_type,
    ]

    if gen_type == "qa":
        cmd.extend(["--num-pairs", str(num_pairs)])

    # Run subprocess in thread to not block event loop
    result = await asyncio.to_thread(subprocess.run, cmd, capture_output=True, text=True)

    if result.returncode != 0:
        if log_fn:
            stderr = result.stderr.strip()
            # Provide helpful error messages
            if "out of memory" in stderr.lower() or "OOM" in stderr:
                await log_fn("    ❌ Out of memory. Try fewer pairs or smaller chunks.")
            elif "timeout" in stderr.lower():
                await log_fn("    ❌ Generation timed out. Try reducing num_pairs.")
            elif "empty" in stderr.lower() or "no content" in stderr.lower():
                await log_fn("    ⚠️ Chunk has no usable content, skipping.")
            else:
                await log_fn(f"    ❌ Generation error: {stderr[:150]}")
        return None

    chunk_name = Path(chunk_file).stem
    suffix = TYPE_SUFFIXES.get(gen_type, "_generated")
    generated_path = Path(f"data/generated/{chunk_name}{suffix}.json")

    if generated_path.exists():
        return generated_path

    for f in Path("data/generated").glob(f"{chunk_name}*.json"):
        return f

    return None


async def curate_content_async(
    json_file: Path,
    config_path: Path,
    threshold: float = 7.5,
    log_fn: Optional[Callable] = None,
) -> Path:
    """Curate content using Llama-as-judge (async)."""
    if log_fn:
        await log_fn(f"  🔍 Curating: {json_file.name}")

    cmd = [
        "synthetic-data-kit",
        "-c",
        str(config_path),
        "curate",
        str(json_file),
        "--threshold",
        str(threshold),
    ]

    result = await asyncio.to_thread(subprocess.run, cmd, capture_output=True, text=True)

    if result.returncode != 0:
        if log_fn:
            await log_fn("    ⚠️ Curation error, using original")
        return json_file

    curated_path = Path(f"data/curated/{json_file.stem}_cleaned.json")
    if curated_path.exists():
        return curated_path

    return json_file


async def convert_to_ft_format_async(
    json_file: Path,
    config_path: Path,
    log_fn: Optional[Callable] = None,
) -> Optional[Path]:
    """Convert to fine-tuning format (async)."""
    cmd = [
        "synthetic-data-kit",
        "-c",
        str(config_path),
        "save-as",
        str(json_file),
        "-f",
        "ft",
    ]

    await asyncio.to_thread(subprocess.run, cmd, capture_output=True, text=True)

    ft_path = Path(f"data/final/{json_file.stem}_ft.json")
    if ft_path.exists():
        return ft_path

    for f in Path("data/final").glob(f"{json_file.stem}*.json"):
        return f

    return None


def convert_to_chatml(data: List[Dict]) -> List[Dict]:
    """Convert fine-tuning format to ChatML format."""
    chatml_data = []
    for item in data:
        messages = item.get("messages", [])
        if messages:
            chatml_data.append({"messages": messages})
    return chatml_data


def convert_to_standard(data: List[Dict]) -> List[Dict]:
    """Convert to standard input/output format."""
    standard_data = []
    for item in data:
        messages = item.get("messages", [])
        user_text = ""
        assistant_text = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_text = msg.get("content", "")
            elif msg.get("role") == "assistant":
                assistant_text = msg.get("content", "")
                break
        if user_text and assistant_text:
            standard_data.append({"input": user_text, "output": assistant_text})
    return standard_data


async def run_synthetic_generation(
    page: ft.Page,
    log_view: ft.ListView,
    prog: ft.ProgressBar,
    labels: Dict[str, ft.Text],
    preview_host: ft.ListView,
    cancel_flag: Dict[str, bool],
    sources: List[str],
    gen_type: str = "qa",
    num_pairs: int = 25,
    max_chunks: int = 10,
    curate: bool = False,
    curate_threshold: float = 7.5,
    multimodal: bool = False,
    output_path: str = "synthetic_data.json",
    dataset_format: str = "ChatML",
    model: str = "unsloth/Llama-3.2-3B-Instruct",
) -> None:
    """Run synthetic data generation - adapted from unsloth-synth-test."""
    import time

    # Track timing for estimates
    start_time = time.time()
    chunk_times: List[float] = []

    def format_time(seconds: float) -> str:
        """Format seconds into human-readable string."""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"

    def estimate_remaining(current: int, total: int) -> str:
        """Estimate remaining time based on chunk processing times."""
        if not chunk_times or current >= total:
            return ""
        avg_time = sum(chunk_times) / len(chunk_times)
        remaining = (total - current) * avg_time
        return f" (~{format_time(remaining)} remaining)"

    async def log(msg: str):
        print(f"[synthetic] {msg}")
        try:
            log_view.controls.append(ft.Text(msg, size=12))
            await safe_update(page)
        except Exception as e:
            print(f"[synthetic] UI log error: {e}")

    async def update_progress(current: int, total: int):
        try:
            prog.value = current / total if total > 0 else 0
            # Update progress info with time estimate
            estimate = estimate_remaining(current, total)
            pct = int((current / total) * 100) if total > 0 else 0
            # Update threads label with progress info (reuse existing label)
            labels["threads"].value = f"Progress: {pct}% ({current}/{total} chunks){estimate}"
            await safe_update(page)
        except Exception as e:
            print(f"[synthetic] UI progress error: {e}")

    async def update_stats(sources_done: int, pairs_found: int):
        try:
            labels["threads"].value = f"Sources Processed: {sources_done}"
            labels["pairs"].value = f"Pairs Generated: {pairs_found}"
            await safe_update(page)
        except Exception as e:
            print(f"[synthetic] UI stats error: {e}")

    # Show immediate feedback in logs
    await log("🚀 Starting synthetic data generation")
    await log(f"   Model: {model}")
    await log(f"   Type: {gen_type}")
    await log(f"   Sources: {len(sources)}")
    await log("")
    await log("⏳ Loading model and starting vLLM server...")
    await log("   (This may take 30-60 seconds on first run)")

    # Clear and create output directories
    for folder in ["data/output", "data/generated", "data/curated", "data/final"]:
        if Path(folder).exists():
            shutil.rmtree(folder)
        Path(folder).mkdir(parents=True, exist_ok=True)

    config_path = create_config_file()

    # Initialize generator (run in thread to not block UI)
    try:
        from unsloth.dataprep import SyntheticDataKit

        def load_model():
            gen = SyntheticDataKit.from_pretrained(
                model_name=model,
                max_seq_length=2048,
            )
            gen.prepare_qa_generation(
                output_folder="data",
                temperature=0.7,
                top_p=0.95,
                overlap=64,
                max_generation_tokens=512,
            )
            return gen

        generator = await asyncio.to_thread(load_model)
        await log(f"✅ Model loaded: {model}")
    except Exception as e:
        error_msg = str(e)
        # Provide helpful error messages for common issues
        if "CUDA" in error_msg or "cuda" in error_msg:
            friendly_msg = "GPU/CUDA error. Ensure CUDA is installed and GPU has enough memory."
        elif "out of memory" in error_msg.lower() or "OOM" in error_msg:
            friendly_msg = "Out of memory. Try a smaller model or reduce batch size."
        elif "vllm" in error_msg.lower():
            friendly_msg = "vLLM server error. Ensure vLLM is installed: pip install vllm"
        elif "connection" in error_msg.lower() or "timeout" in error_msg.lower():
            friendly_msg = "Connection error. Check your network and try again."
        elif "not found" in error_msg.lower() or "No module" in error_msg:
            friendly_msg = "Missing dependency. Run: pip install unsloth vllm"
        else:
            friendly_msg = error_msg[:150]

        await log(f"❌ Failed to load model: {friendly_msg}")
        await log(f"   Technical details: {error_msg[:200]}")
        try:
            page.snack_bar = ft.SnackBar(
                ft.Text(f"❌ Model load failed: {friendly_msg}"),
                bgcolor=ft.colors.RED_700,
                duration=8000,
            )
            page.open(page.snack_bar)
            await safe_update(page)
        except Exception:
            pass
        return

    all_ft_files = []
    total_pairs = 0
    sources_processed = 0

    # Calculate total steps for progress
    total_steps = len(sources) * max_chunks  # Approximate
    current_step = 0

    try:
        for idx, source in enumerate(sources):
            if cancel_flag.get("cancelled"):
                await log("🛑 Cancelled by user")
                break

            source_str = str(source)
            source_name = source_str if is_url(source_str) else Path(source_str).name

            await log(f"\n{'=' * 40}")
            await log(f"📁 Processing: {source_name}")
            await log(f"{'=' * 40}")

            # Ingest source (async)
            txt_file = await ingest_source_async(source_str, config_path, multimodal, log)
            if not txt_file:
                await log("  ⚠️ Failed to ingest, skipping...")
                continue

            # Chunk the document
            await log("  📝 Chunking document...")
            chunk_files = await asyncio.to_thread(generator.chunk_data, str(txt_file))

            if not chunk_files:
                await log("  📊 File too small to chunk, using as single chunk")
                chunk_files = [str(txt_file)]
            else:
                await log(f"  📊 Created {len(chunk_files)} chunks")

            # Process chunks
            chunks_to_process = chunk_files[:max_chunks]
            await log(f"  🔄 Processing {len(chunks_to_process)} chunks...")

            for i, chunk_file in enumerate(chunks_to_process):
                if cancel_flag.get("cancelled"):
                    break

                chunk_start = time.time()
                current_step += 1
                await update_progress(current_step, total_steps)

                await log(f"\n  [{i + 1}/{len(chunks_to_process)}] Generating {gen_type}...")

                # Generate content (async)
                gen_file = await generate_content_async(chunk_file, config_path, gen_type, num_pairs, log)
                if not gen_file:
                    await log("    ❌ Generation failed for this chunk")
                    continue

                # Optionally curate (async)
                if curate:
                    gen_file = await curate_content_async(gen_file, config_path, curate_threshold, log)

                # Convert to fine-tuning format (async)
                ft_file = await convert_to_ft_format_async(gen_file, config_path, log)
                if ft_file:
                    all_ft_files.append(ft_file)
                    total_pairs += num_pairs  # Approximate count
                    chunk_elapsed = time.time() - chunk_start
                    chunk_times.append(chunk_elapsed)
                    await log(f"  ✅ Generated: {ft_file.name} ({format_time(chunk_elapsed)})")
                    await update_stats(sources_processed, total_pairs)

                await asyncio.sleep(0.5)  # Brief pause, non-blocking

            sources_processed += 1
            await update_stats(sources_processed, total_pairs)

        # Combine all datasets
        if all_ft_files:
            await log(f"\n📦 Combining {len(all_ft_files)} files into final dataset...")

            import pandas as pd

            conversations = pd.concat([pd.read_json(f) for f in all_ft_files]).reset_index(drop=True)

            combined_data = conversations.to_dict(orient="records")
            if dataset_format == "Standard":
                final_data = convert_to_standard(combined_data)
            else:
                final_data = convert_to_chatml(combined_data)

            total_pairs = len(final_data)
            pairs_output = dataset_format.lower() == "standard"

            # Write to output file
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(final_data, f, ensure_ascii=False, indent=2)

            await log(f"💾 Saved to: {output_path}")

            # Save to database
            try:
                source_details = f"sources={len(sources)}, type={gen_type}, model={model}"
                if pairs_output:
                    await asyncio.to_thread(
                        lambda: save_scrape_to_db(
                            source="synthetic",
                            pairs=final_data,
                            source_details=source_details,
                            dataset_format=dataset_format,
                        )
                    )
                else:
                    await asyncio.to_thread(
                        lambda: save_chatml_to_db(
                            source="synthetic",
                            conversations=final_data,
                            source_details=source_details,
                        )
                    )
                await log("📊 Saved to database")
            except Exception as db_err:
                await log(f"⚠️ Database save warning: {db_err}")

            total_elapsed = time.time() - start_time
            await log(f"\n✨ Done! Generated {total_pairs} {'pairs' if pairs_output else 'conversations'} in {format_time(total_elapsed)}")
            await update_stats(sources_processed, total_pairs)

            # Update labels to match other scrapers
            labels["pairs"].value = f"{'Pairs' if pairs_output else 'Conversations'} Found: {total_pairs}"
            labels["threads"].value = f"Sources: {sources_processed} | Time: {format_time(total_elapsed)}"

            # Build preview pairs for display
            preview_pairs = []
            for item in final_data[:10]:
                if pairs_output:
                    user_msg = item.get("input", "") or ""
                    asst_msg = item.get("output", "") or ""
                else:
                    msgs = item.get("messages", [])
                    user_msg = ""
                    asst_msg = ""
                    for m in msgs:
                        if not isinstance(m, dict):
                            continue
                        role = m.get("role")
                        text = (m.get("content") or "").strip()
                        if role == "user" and not user_msg:
                            user_msg = text
                        elif role == "assistant" and user_msg and not asst_msg:
                            asst_msg = text
                            break
                if user_msg or asst_msg:
                    preview_pairs.append((user_msg, asst_msg))

            # Populate preview grid (same style as other scrapers)
            try:
                preview_host.controls.clear()
                if preview_pairs:
                    lfx, rfx = compute_two_col_flex(preview_pairs)
                    hdr_left = "Input" if pairs_output else "User"
                    hdr_right = "Output" if pairs_output else "Assistant"
                    preview_host.controls.append(two_col_header(hdr_left, hdr_right, left_flex=lfx, right_flex=rfx))
                    for a, b in preview_pairs:
                        preview_host.controls.append(two_col_row(a, b, lfx, rfx))
                await safe_update(page)
            except Exception as e:
                await log(f"⚠️ Failed to render preview: {e}")

            # Show success snackbar
            page.snack_bar = ft.SnackBar(ft.Text("Generation complete! ✨"))
            page.open(page.snack_bar)
            await safe_update(page)
        else:
            await log("\n❌ No content was generated!")

    finally:
        await log("\n🧹 Cleaning up...")
        try:
            generator.cleanup()
        except Exception:
            pass

    await update_progress(1, 1)
    await update_stats(sources_processed, total_pairs)
