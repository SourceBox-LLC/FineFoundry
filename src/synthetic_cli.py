#!/usr/bin/env python3
"""CLI for synthetic data generation.

Generate synthetic training data from documents, URLs, or other sources
using Unsloth's SyntheticDataKit.

Examples:
    # Generate Q&A pairs from a PDF
    python src/synthetic_cli.py --source document.pdf --output qa_pairs.json

    # Generate from multiple sources
    python src/synthetic_cli.py --source doc1.pdf --source doc2.txt --source https://example.com

    # Generate Chain-of-Thought data with curation
    python src/synthetic_cli.py --source paper.pdf --type cot --curate --threshold 8.0

    # Use a specific model
    python src/synthetic_cli.py --source data.txt --model unsloth/Llama-3.2-1B-Instruct
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from helpers.synthetic import (
    SUPPORTED_EXTENSIONS,
    TYPE_SUFFIXES,
    is_url,
    create_config_file,
    convert_to_chatml,
    convert_to_standard,
)
from helpers.scrape_db import save_scrape_to_db, save_chatml_to_db


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


def log(msg: str, quiet: bool = False, verbose: bool = False, level: str = "info") -> None:
    """Print log message unless quiet mode.

    Args:
        msg: Message to print
        quiet: If True, suppress all output
        verbose: If True, show debug-level messages
        level: Message level - 'info', 'debug', 'warn', 'error'
    """
    if quiet:
        return
    if level == "debug" and not verbose:
        return
    print(msg)


def debug(msg: str, verbose: bool = False) -> None:
    """Print debug message only in verbose mode."""
    if verbose:
        print(f"  [DEBUG] {msg}")


def validate_source(source: str) -> bool:
    """Validate that a source is a supported file or URL."""
    if is_url(source):
        return True
    path = Path(source)
    if not path.exists():
        print(f"Error: File not found: {source}", file=sys.stderr)
        return False
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        print(f"Error: Unsupported file type: {path.suffix}", file=sys.stderr)
        print(f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}", file=sys.stderr)
        return False
    return True


def load_config_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file.

    Config file format:
        sources:
          - document.pdf
          - https://example.com/article
        output: synthetic_data.json
        type: qa
        num_pairs: 25
        max_chunks: 10
        model: unsloth/Llama-3.2-3B-Instruct
        curate: false
        threshold: 7.5
        format: chatml
        multimodal: false
        save_to_db: true
    """
    path = Path(config_path)
    if not path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config if config else {}
    except yaml.YAMLError as e:
        print(f"Error parsing config file: {e}", file=sys.stderr)
        sys.exit(1)


def create_progress_bar(total: int, desc: str, quiet: bool = False):
    """Create a progress bar if tqdm is available and not in quiet mode."""
    if quiet:
        return None
    try:
        from tqdm import tqdm

        return tqdm(total=total, desc=desc, unit="chunk", leave=True)
    except ImportError:
        return None


def check_vllm_running(port: int = 8000) -> bool:
    """Check if a vLLM server is already running on the specified port."""
    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex(("localhost", port))
            return result == 0
    except Exception:
        return False


def deduplicate_data(data: List[Dict[str, Any]], dataset_format: str) -> List[Dict[str, Any]]:
    """Remove duplicate entries based on input/user content.
    
    Args:
        data: List of conversation/pair dictionaries
        dataset_format: 'chatml' or 'standard'
        
    Returns:
        Deduplicated list
    """
    seen = set()
    unique = []
    
    for item in data:
        # Extract the key for deduplication
        if dataset_format.lower() == "standard":
            key = item.get("input", "")
        else:
            # ChatML format - use first user message
            messages = item.get("messages", [])
            key = ""
            for msg in messages:
                if msg.get("role") == "user":
                    key = msg.get("content", "")
                    break
        
        # Normalize key
        key = key.strip().lower()
        
        if key and key not in seen:
            seen.add(key)
            unique.append(item)
    
    return unique


def load_existing_data(output_path: str, output_type: str) -> List[Dict[str, Any]]:
    """Load existing data from output file for resume functionality.
    
    Args:
        output_path: Path to existing output
        output_type: 'json', 'hf', or 'parquet'
        
    Returns:
        List of existing data or empty list
    """
    try:
        if output_type == "json":
            if Path(output_path).exists():
                with open(output_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        elif output_type == "hf":
            if Path(output_path).exists():
                try:
                    from datasets import load_from_disk
                    ds = load_from_disk(output_path)
                    if hasattr(ds, "get"):
                        ds = ds.get("train", ds)
                    return list(ds)
                except Exception:
                    pass
        elif output_type == "parquet":
            if Path(output_path).exists():
                try:
                    import pandas as pd
                    df = pd.read_parquet(output_path)
                    return df.to_dict("records")
                except Exception:
                    pass
    except Exception:
        pass
    return []


def save_output(
    data: List[Dict[str, Any]],
    output_path: str,
    output_type: str,
    dataset_format: str,
    quiet: bool = False,
) -> bool:
    """Save data to the specified output format.
    
    Args:
        data: List of conversation/pair dictionaries
        output_path: Path to save output
        output_type: 'json', 'hf', or 'parquet'
        dataset_format: 'chatml' or 'standard'
        quiet: Suppress output
        
    Returns:
        True if successful
    """
    try:
        if output_type == "json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            log(f"💾 Saved JSON to: {output_path}", quiet)
            
        elif output_type == "hf":
            try:
                from datasets import Dataset, DatasetDict
                ds = Dataset.from_list(data)
                dd = DatasetDict({"train": ds})
                Path(output_path).mkdir(parents=True, exist_ok=True)
                dd.save_to_disk(output_path)
                log(f"💾 Saved HuggingFace dataset to: {output_path}", quiet)
            except ImportError:
                log("⚠️ datasets library not installed, falling back to JSON", quiet)
                json_path = output_path if output_path.endswith(".json") else f"{output_path}.json"
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                log(f"💾 Saved JSON to: {json_path}", quiet)
                
        elif output_type == "parquet":
            try:
                import pandas as pd
                df = pd.DataFrame(data)
                parquet_path = output_path if output_path.endswith(".parquet") else f"{output_path}.parquet"
                df.to_parquet(parquet_path, index=False)
                log(f"💾 Saved Parquet to: {parquet_path}", quiet)
            except ImportError:
                log("⚠️ pandas not installed, falling back to JSON", quiet)
                json_path = output_path if output_path.endswith(".json") else f"{output_path}.json"
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                log(f"💾 Saved JSON to: {json_path}", quiet)
                
        return True
    except Exception as e:
        print(f"Error saving output: {e}", file=sys.stderr)
        return False


def push_to_hf_hub(
    data: List[Dict[str, Any]],
    repo_id: str,
    dataset_format: str,
    private: bool = False,
    quiet: bool = False,
) -> bool:
    """Push dataset to HuggingFace Hub.
    
    Args:
        data: List of conversation/pair dictionaries
        repo_id: HuggingFace repo ID (e.g., 'username/dataset-name')
        dataset_format: 'chatml' or 'standard'
        private: Whether to create a private repo
        quiet: Suppress output
        
    Returns:
        True if successful
    """
    try:
        from datasets import Dataset
        
        log(f"🚀 Pushing to HuggingFace Hub: {repo_id}", quiet)
        
        # Create dataset
        ds = Dataset.from_list(data)
        
        # Push to hub
        ds.push_to_hub(repo_id, private=private)
        
        log(f"✅ Successfully pushed to: https://huggingface.co/datasets/{repo_id}", quiet)
        return True
        
    except ImportError:
        print("Error: 'datasets' and 'huggingface_hub' libraries required for Hub push", file=sys.stderr)
        print("Install with: pip install datasets huggingface_hub", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Error pushing to Hub: {e}", file=sys.stderr)
        return False


def compute_dataset_stats(
    data: List[Dict[str, Any]],
    dataset_format: str,
) -> Dict[str, Any]:
    """Compute statistics for a dataset.
    
    Args:
        data: List of conversation/pair dictionaries
        dataset_format: 'chatml' or 'standard'
        
    Returns:
        Dictionary with statistics
    """
    if not data:
        return {"count": 0}
    
    stats = {
        "count": len(data),
        "input_lengths": [],
        "output_lengths": [],
        "total_chars": 0,
    }
    
    for item in data:
        if dataset_format.lower() == "standard":
            inp = item.get("input", "")
            out = item.get("output", "")
        else:
            # ChatML format
            messages = item.get("messages", [])
            inp = ""
            out = ""
            for msg in messages:
                if msg.get("role") == "user" and not inp:
                    inp = msg.get("content", "")
                elif msg.get("role") == "assistant" and not out:
                    out = msg.get("content", "")
        
        inp_len = len(inp)
        out_len = len(out)
        stats["input_lengths"].append(inp_len)
        stats["output_lengths"].append(out_len)
        stats["total_chars"] += inp_len + out_len
    
    # Compute aggregates
    if stats["input_lengths"]:
        stats["avg_input_len"] = sum(stats["input_lengths"]) / len(stats["input_lengths"])
        stats["max_input_len"] = max(stats["input_lengths"])
        stats["min_input_len"] = min(stats["input_lengths"])
        stats["avg_output_len"] = sum(stats["output_lengths"]) / len(stats["output_lengths"])
        stats["max_output_len"] = max(stats["output_lengths"])
        stats["min_output_len"] = min(stats["output_lengths"])
    
    # Estimate tokens (rough: ~4 chars per token)
    stats["estimated_tokens"] = stats["total_chars"] // 4
    
    # Remove raw lists for cleaner output
    del stats["input_lengths"]
    del stats["output_lengths"]
    
    return stats


def run_with_retry(
    func,
    max_retries: int = 3,
    delay: float = 2.0,
    quiet: bool = False,
):
    """Run a function with retry logic for network errors.
    
    Args:
        func: Function to call (should return result or raise exception)
        max_retries: Maximum number of retry attempts
        delay: Delay between retries in seconds
        quiet: Suppress output
        
    Returns:
        Function result or None on failure
    """
    import time
    
    last_error = None
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_error = e
            error_msg = str(e).lower()
            # Check if it's a retryable error
            retryable = any(x in error_msg for x in [
                "connection", "timeout", "network", "temporary",
                "429", "503", "502", "504", "rate limit"
            ])
            
            if retryable and attempt < max_retries - 1:
                wait_time = delay * (2 ** attempt)  # Exponential backoff
                if not quiet:
                    log(f"  ⚠️ Retry {attempt + 1}/{max_retries} in {wait_time:.1f}s: {str(e)[:50]}", quiet)
                time.sleep(wait_time)
            else:
                break
    
    if last_error:
        raise last_error
    return None


def save_progress(
    progress_file: str,
    sources_completed: List[str],
    chunks_completed: Dict[str, List[int]],
    data_so_far: List[Dict[str, Any]],
) -> None:
    """Save generation progress to a file for resume capability.
    
    Args:
        progress_file: Path to progress file
        sources_completed: List of fully processed source paths
        chunks_completed: Dict mapping source -> list of completed chunk indices
        data_so_far: Data generated so far
    """
    progress = {
        "sources_completed": sources_completed,
        "chunks_completed": chunks_completed,
        "data_count": len(data_so_far),
        "timestamp": time.time(),
    }
    try:
        with open(progress_file, "w", encoding="utf-8") as f:
            json.dump(progress, f, indent=2)
    except Exception:
        pass  # Non-critical, don't fail on progress save


def load_progress(progress_file: str) -> Optional[Dict[str, Any]]:
    """Load generation progress from a file.
    
    Args:
        progress_file: Path to progress file
        
    Returns:
        Progress dict or None if not found/invalid
    """
    try:
        if Path(progress_file).exists():
            with open(progress_file, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return None


def clear_progress(progress_file: str) -> None:
    """Remove progress file after successful completion."""
    try:
        if Path(progress_file).exists():
            Path(progress_file).unlink()
    except Exception:
        pass


def run_subprocess(cmd: List[str], description: str, quiet: bool = False) -> bool:
    """Run a subprocess command and return success status."""
    if not quiet:
        log(f"  Running: {' '.join(cmd[:4])}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error in {description}: {result.stderr[:200]}", file=sys.stderr)
        return False
    return True


def ingest_source(source: str, config_path: Path, multimodal: bool, quiet: bool) -> Optional[Path]:
    """Ingest a source and return the output text file path."""
    from urllib.parse import urlparse
    import subprocess

    log(f"📄 Ingesting: {source}", quiet)

    cmd = ["synthetic-data-kit", "-c", str(config_path), "ingest", source]
    if multimodal:
        cmd.append("--multimodal")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ❌ Ingestion failed: {result.stderr[:150]}", file=sys.stderr)
        return None

    output_dir = Path("data/output")
    if is_url(source):
        parsed = urlparse(source)
        base_name = parsed.netloc.replace(".", "_")
    else:
        base_name = Path(source).stem.replace(" ", "_").replace("-", "_")

    patterns = [f"*{base_name}*", "*.txt", "*.lance"]
    for pattern in patterns:
        files = list(output_dir.glob(pattern))
        if files:
            return files[0]

    return None


def generate_content(
    chunk_file: str,
    config_path: Path,
    gen_type: str,
    num_pairs: int,
    quiet: bool,
) -> Optional[Path]:
    """Generate content from a chunk file."""
    import subprocess

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

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        if not quiet:
            print(f"    ⚠️ Generation warning: {result.stderr[:100]}")
        return None

    chunk_name = Path(chunk_file).stem
    suffix = TYPE_SUFFIXES.get(gen_type, "_generated")
    generated_path = Path(f"data/generated/{chunk_name}{suffix}.json")

    if generated_path.exists():
        return generated_path

    for f in Path("data/generated").glob(f"{chunk_name}*.json"):
        return f

    return None


def curate_content(json_file: Path, config_path: Path, threshold: float, quiet: bool) -> Path:
    """Curate content using quality threshold."""
    import subprocess

    log(f"  🔍 Curating with threshold {threshold}...", quiet)

    cmd = [
        "synthetic-data-kit",
        "-c",
        str(config_path),
        "curate",
        str(json_file),
        "--threshold",
        str(threshold),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        if not quiet:
            print(f"    ⚠️ Curation warning: {result.stderr[:100]}")
        return json_file

    curated_path = Path(f"data/curated/{json_file.stem}_curated.json")
    if curated_path.exists():
        return curated_path

    return json_file


def convert_to_ft_format(json_file: Path, config_path: Path, quiet: bool) -> Optional[List[dict]]:
    """Convert generated JSON to fine-tuning format and return the data."""
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle different output formats from synthetic-data-kit
        qa_pairs = []
        if isinstance(data, dict):
            # Format: {"summary": "...", "qa_pairs": [...]}
            if "qa_pairs" in data:
                for pair in data["qa_pairs"]:
                    qa_pairs.append(
                        {
                            "messages": [
                                {"role": "user", "content": pair.get("question", "")},
                                {"role": "assistant", "content": pair.get("answer", "")},
                            ]
                        }
                    )
            # Format: {"instruction": "...", "output": "..."}
            elif "instruction" in data:
                qa_pairs.append(
                    {
                        "messages": [
                            {"role": "user", "content": data.get("instruction", "")},
                            {"role": "assistant", "content": data.get("output", "")},
                        ]
                    }
                )
        elif isinstance(data, list):
            # Format: [{"question": "...", "answer": "..."}, ...]
            for item in data:
                if "question" in item and "answer" in item:
                    qa_pairs.append(
                        {
                            "messages": [
                                {"role": "user", "content": item["question"]},
                                {"role": "assistant", "content": item["answer"]},
                            ]
                        }
                    )
                elif "instruction" in item:
                    qa_pairs.append(
                        {
                            "messages": [
                                {"role": "user", "content": item.get("instruction", "")},
                                {"role": "assistant", "content": item.get("output", "")},
                            ]
                        }
                    )
                elif "messages" in item:
                    qa_pairs.append(item)

        return qa_pairs if qa_pairs else None
    except Exception as e:
        if not quiet:
            print(f"    ⚠️ Conversion error: {e}")
        return None


def run_generation(
    sources: List[str],
    output_path: str,
    gen_type: str = "qa",
    num_pairs: int = 25,
    max_chunks: int = 10,
    curate: bool = False,
    curate_threshold: float = 7.5,
    multimodal: bool = False,
    dataset_format: str = "chatml",
    output_type: str = "json",
    model: str = "unsloth/Llama-3.2-3B-Instruct",
    save_to_db: bool = True,
    quiet: bool = False,
    verbose: bool = False,
    keep_server: bool = False,
    dedupe: bool = False,
    resume: bool = False,
    show_stats: bool = False,
    push_to_hub: Optional[str] = None,
    private: bool = False,
) -> bool:
    """Run the full synthetic data generation pipeline.

    Args:
        sources: List of file paths or URLs to process
        output_path: Path to save the output JSON file
        gen_type: Generation type - 'qa', 'cot', or 'summary'
        num_pairs: Number of pairs to generate per chunk
        max_chunks: Maximum chunks to process per source
        curate: Enable quality curation
        curate_threshold: Quality threshold for curation (1-10)
        multimodal: Enable multimodal processing
        dataset_format: Output format - 'chatml' or 'standard'
        output_type: Output type - 'json', 'hf', or 'parquet'
        model: Model name to use for generation
        save_to_db: Save results to FineFoundry database
        quiet: Suppress all output except errors
        verbose: Show detailed debug output
        keep_server: Keep vLLM server running after generation (for batch runs)
        dedupe: Remove duplicate entries based on input text
        resume: Resume from previous run (load existing output and append)
        show_stats: Show dataset statistics after generation
        push_to_hub: HuggingFace Hub repo ID to push to (optional)
        private: Make Hub repo private (use with push_to_hub)
    """
    start_time = time.time()

    log("🚀 Starting synthetic data generation", quiet)
    log(f"   Model: {model}", quiet)
    log(f"   Type: {gen_type}", quiet)
    log(f"   Sources: {len(sources)}", quiet)
    if verbose:
        debug(f"Output: {output_path}", verbose)
        debug(f"Output type: {output_type}", verbose)
        debug(f"Num pairs per chunk: {num_pairs}", verbose)
        debug(f"Max chunks per source: {max_chunks}", verbose)
        debug(f"Curate: {curate} (threshold: {curate_threshold})", verbose)
        debug(f"Format: {dataset_format}", verbose)
        debug(f"Multimodal: {multimodal}", verbose)
        debug(f"Save to DB: {save_to_db}", verbose)
        debug(f"Dedupe: {dedupe}", verbose)
        debug(f"Resume: {resume}", verbose)
    log("", quiet)

    # Progress file for mid-run resume
    progress_file = f"{output_path}.progress"
    
    # Load existing data and progress if resuming
    existing_data: List[Dict[str, Any]] = []
    progress_state: Optional[Dict[str, Any]] = None
    sources_completed: List[str] = []
    chunks_completed: Dict[str, List[int]] = {}
    
    if resume:
        existing_data = load_existing_data(output_path, output_type)
        progress_state = load_progress(progress_file)
        
        if progress_state:
            sources_completed = progress_state.get("sources_completed", [])
            chunks_completed = progress_state.get("chunks_completed", {})
            log(f"♻️  Resuming: {len(sources_completed)} sources done, {len(existing_data)} entries", quiet)
            debug(f"Progress: sources={sources_completed}, chunks={chunks_completed}", verbose)
        elif existing_data:
            log(f"♻️  Resuming with {len(existing_data)} existing entries", quiet)
            debug(f"Loaded {len(existing_data)} entries from {output_path}", verbose)
        else:
            log("📝 No existing data found, starting fresh", quiet)

    # Clear and create output directories
    for folder in ["data/output", "data/generated", "data/curated", "data/final"]:
        if Path(folder).exists():
            shutil.rmtree(folder)
        Path(folder).mkdir(parents=True, exist_ok=True)

    config_path = create_config_file()

    # Check if vLLM server is already running
    server_was_running = check_vllm_running()
    if server_was_running:
        log("♻️  Reusing existing vLLM server on port 8000", quiet)
        debug("Server already running, skipping model load", verbose)
    else:
        log("⏳ Loading model and starting vLLM server...", quiet)
        log("   (This may take 30-60 seconds on first run)", quiet)

    try:
        from unsloth.dataprep import SyntheticDataKit

        generator = SyntheticDataKit.from_pretrained(
            model_name=model,
            max_seq_length=2048,
        )
        generator.prepare_qa_generation(
            output_folder="data",
            temperature=0.7,
            top_p=0.95,
            overlap=64,
            max_generation_tokens=512,
        )
        if not server_was_running:
            log(f"✅ Model loaded: {model}", quiet)
    except Exception as e:
        error_msg = str(e)
        if "CUDA" in error_msg or "cuda" in error_msg:
            print("❌ GPU/CUDA error. Ensure CUDA is installed and GPU has enough memory.", file=sys.stderr)
        elif "out of memory" in error_msg.lower():
            print("❌ Out of memory. Try a smaller model.", file=sys.stderr)
        elif "not found" in error_msg.lower() or "No module" in error_msg:
            print("❌ Missing dependency. Run: pip install unsloth vllm", file=sys.stderr)
        else:
            print(f"❌ Failed to load model: {error_msg[:200]}", file=sys.stderr)
        return False

    all_conversations = []
    total_pairs = 0
    sources_processed = 0

    try:
        for idx, source in enumerate(sources):
            source_name = source if is_url(source) else Path(source).name

            # Skip already completed sources when resuming
            if source in sources_completed:
                log(f"\n⏭️  Skipping completed source [{idx + 1}/{len(sources)}]: {source_name}", quiet)
                continue

            log(f"\n{'=' * 40}", quiet)
            log(f"📁 Processing [{idx + 1}/{len(sources)}]: {source_name}", quiet)
            log(f"{'=' * 40}", quiet)

            # Ingest source
            txt_file = ingest_source(source, config_path, multimodal, quiet)
            if not txt_file:
                log("  ⚠️ Failed to ingest, skipping...", quiet)
                continue

            # Chunk the document
            log("  📝 Chunking document...", quiet)
            chunk_files = generator.chunk_data(str(txt_file))

            if not chunk_files:
                log("  📊 File too small to chunk, using as single chunk", quiet)
                chunk_files = [str(txt_file)]
            else:
                log(f"  📊 Created {len(chunk_files)} chunks", quiet)

            # Process chunks
            chunks_to_process = chunk_files[:max_chunks]
            completed_for_source = chunks_completed.get(source, [])
            log(f"  🔄 Processing {len(chunks_to_process)} chunks...", quiet)

            # Create progress bar for chunks
            pbar = create_progress_bar(len(chunks_to_process), f"  {source_name[:20]}", quiet)

            for i, chunk_file in enumerate(chunks_to_process):
                # Skip already completed chunks when resuming
                if i in completed_for_source:
                    debug(f"Skipping completed chunk {i + 1}", verbose)
                    if pbar:
                        pbar.update(1)
                    continue
                    
                chunk_start = time.time()
                debug(f"Processing chunk {i + 1}: {chunk_file}", verbose)

                if not pbar:
                    log(f"\n  [{i + 1}/{len(chunks_to_process)}] Generating {gen_type}...", quiet)

                # Generate content
                gen_file = generate_content(chunk_file, config_path, gen_type, num_pairs, quiet)
                if not gen_file:
                    if pbar:
                        pbar.set_postfix({"status": "failed"})
                    else:
                        log("    ❌ Generation failed for this chunk", quiet)
                    if pbar:
                        pbar.update(1)
                    continue

                # Optionally curate
                if curate:
                    debug(f"Curating with threshold {curate_threshold}", verbose)
                    gen_file = curate_content(gen_file, config_path, curate_threshold, quiet)

                # Convert to fine-tuning format
                ft_data = convert_to_ft_format(gen_file, config_path, quiet)
                if ft_data:
                    all_conversations.extend(ft_data)
                    chunk_elapsed = time.time() - chunk_start
                    if pbar:
                        pbar.set_postfix({"pairs": len(ft_data), "time": format_time(chunk_elapsed)})
                    else:
                        log(f"  ✅ Generated {len(ft_data)} pairs ({format_time(chunk_elapsed)})", quiet)
                    
                    # Save progress after each chunk
                    if source not in chunks_completed:
                        chunks_completed[source] = []
                    chunks_completed[source].append(i)
                    save_progress(progress_file, sources_completed, chunks_completed, all_conversations)

                if pbar:
                    pbar.update(1)

            if pbar:
                pbar.close()
            sources_processed += 1
            
            # Mark source as completed
            sources_completed.append(source)
            save_progress(progress_file, sources_completed, chunks_completed, all_conversations)

        # Combine all datasets
        if all_conversations or existing_data:
            new_count = len(all_conversations)
            log(f"\n📦 Combining {new_count} new conversations into final dataset...", quiet)

            combined_data = all_conversations

            if dataset_format.lower() == "standard":
                final_data = convert_to_standard(combined_data)
            else:
                final_data = convert_to_chatml(combined_data)

            # Merge with existing data if resuming
            if existing_data:
                final_data = existing_data + final_data
                log(f"   Merged with {len(existing_data)} existing entries", quiet)

            # Deduplicate if requested
            if dedupe:
                before_count = len(final_data)
                final_data = deduplicate_data(final_data, dataset_format)
                removed = before_count - len(final_data)
                if removed > 0:
                    log(f"🔄 Removed {removed} duplicate entries", quiet)

            total_pairs = len(final_data)
            pairs_output = dataset_format.lower() == "standard"

            # Save to output file using appropriate format
            save_output(final_data, output_path, output_type, dataset_format, quiet)

            # Save to database
            if save_to_db:
                try:
                    source_details = f"sources={len(sources)}, type={gen_type}, model={model}"
                    if pairs_output:
                        save_scrape_to_db(
                            source="synthetic",
                            pairs=final_data,
                            source_details=source_details,
                            dataset_format=dataset_format,
                        )
                    else:
                        save_chatml_to_db(
                            source="synthetic",
                            conversations=final_data,
                            source_details=source_details,
                        )
                    log("📊 Saved to database", quiet)
                except Exception as db_err:
                    log(f"⚠️ Database save warning: {db_err}", quiet)

            total_elapsed = time.time() - start_time
            log(
                f"\n✨ Done! Generated {total_pairs} {'pairs' if pairs_output else 'conversations'} in {format_time(total_elapsed)}",
                quiet,
            )

            # Show dataset statistics if requested
            stats = None
            if show_stats:
                stats = compute_dataset_stats(final_data, dataset_format)
                log("\n📈 Dataset Statistics:", quiet)
                log(f"   Total entries: {stats['count']}", quiet)
                log(f"   Estimated tokens: {stats.get('estimated_tokens', 0):,}", quiet)
                log(f"   Total characters: {stats.get('total_chars', 0):,}", quiet)
                if stats.get('avg_input_len'):
                    log(f"   Avg input length: {stats['avg_input_len']:.0f} chars", quiet)
                    log(f"   Avg output length: {stats['avg_output_len']:.0f} chars", quiet)
                    log(f"   Input range: {stats['min_input_len']}-{stats['max_input_len']} chars", quiet)
                    log(f"   Output range: {stats['min_output_len']}-{stats['max_output_len']} chars", quiet)

            # Push to HuggingFace Hub if requested
            if push_to_hub:
                hub_result = push_to_hf_hub(
                    final_data, push_to_hub, dataset_format, private, quiet
                )
                if not hub_result:
                    log("⚠️ Hub push failed, but local save succeeded", quiet)

            # Print summary in quiet mode too
            if quiet:
                summary = {
                    "success": True,
                    "output": output_path,
                    "pairs": total_pairs,
                    "sources": sources_processed,
                    "elapsed_seconds": round(total_elapsed, 2),
                }
                if stats:
                    summary["stats"] = stats
                if push_to_hub:
                    summary["hub_repo"] = push_to_hub
                print(json.dumps(summary))

            # Clear progress file on successful completion
            clear_progress(progress_file)
            return True
        else:
            print("❌ No content was generated!", file=sys.stderr)
            return False

    finally:
        if keep_server:
            log("\n💡 Keeping vLLM server running (use --no-keep-server to stop)", quiet)
            debug("Server kept alive for subsequent runs", verbose)
        else:
            log("\n🧹 Cleaning up...", quiet)
            try:
                generator.cleanup()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data from documents and URLs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --source document.pdf --output qa_pairs.json
  %(prog)s --source doc1.pdf --source doc2.txt --type qa --num-pairs 50
  %(prog)s --source paper.pdf --type cot --curate --threshold 8.0
  %(prog)s --source https://example.com/article --format standard
        """,
    )

    # Source arguments (required unless using config file)
    parser.add_argument(
        "--source",
        "-s",
        action="append",
        help="Source file or URL (can be specified multiple times)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="synthetic_data.json",
        help="Output JSON file path (default: synthetic_data.json)",
    )

    # Generation options
    parser.add_argument(
        "--type",
        "-t",
        choices=["qa", "cot", "summary"],
        default="qa",
        help="Generation type: qa (Q&A pairs), cot (chain-of-thought), summary (default: qa)",
    )
    parser.add_argument(
        "--num-pairs",
        "-n",
        type=int,
        default=25,
        help="Number of pairs to generate per chunk (default: 25)",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=10,
        help="Maximum chunks to process per source (default: 10)",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="unsloth/Llama-3.2-3B-Instruct",
        help="Model to use for generation (default: unsloth/Llama-3.2-3B-Instruct)",
    )

    # Quality options
    parser.add_argument(
        "--curate",
        action="store_true",
        help="Enable quality curation using Llama-as-judge",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=7.5,
        help="Curation quality threshold 1-10 (default: 7.5)",
    )

    # Format options
    parser.add_argument(
        "--format",
        "-f",
        choices=["chatml", "standard"],
        default="chatml",
        help="Output format: chatml (messages) or standard (input/output) (default: chatml)",
    )
    parser.add_argument(
        "--output-type",
        choices=["json", "hf", "parquet"],
        default="json",
        help="Output type: json (file), hf (HuggingFace datasets dir), parquet (default: json)",
    )
    parser.add_argument(
        "--multimodal",
        action="store_true",
        help="Enable multimodal processing for images in documents",
    )

    # Data processing options
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Remove duplicate pairs based on input text",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous run (loads existing output and appends new data)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show dataset statistics (token counts, length distribution)",
    )

    # HuggingFace Hub options
    parser.add_argument(
        "--push-to-hub",
        type=str,
        metavar="REPO_ID",
        help="Push dataset to HuggingFace Hub (e.g., 'username/dataset-name')",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the HuggingFace Hub repo private (use with --push-to-hub)",
    )

    # Database options
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Don't save results to the FineFoundry database",
    )

    # Output options
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Quiet mode - only output JSON summary on success",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose mode - show detailed debug output",
    )

    # Config file option
    parser.add_argument(
        "--config",
        type=str,
        help="Load options from a YAML config file",
    )

    # Performance options
    parser.add_argument(
        "--keep-server",
        action="store_true",
        help="Keep vLLM server running after generation (faster for batch runs)",
    )

    args = parser.parse_args()

    # Load config file if provided
    config = {}
    if args.config:
        config = load_config_file(args.config)
        if args.verbose:
            print(f"[DEBUG] Loaded config from {args.config}: {config}")

    # Merge config with command-line args (CLI takes precedence)
    sources = args.source if args.source else config.get("sources", [])
    output_path = args.output if args.output != "synthetic_data.json" else config.get("output", args.output)
    gen_type = args.type if args.type != "qa" else config.get("type", args.type)
    num_pairs = args.num_pairs if args.num_pairs != 25 else config.get("num_pairs", args.num_pairs)
    max_chunks = args.max_chunks if args.max_chunks != 10 else config.get("max_chunks", args.max_chunks)
    model = args.model if args.model != "unsloth/Llama-3.2-3B-Instruct" else config.get("model", args.model)
    curate = args.curate or config.get("curate", False)
    threshold = args.threshold if args.threshold != 7.5 else config.get("threshold", args.threshold)
    dataset_format = args.format if args.format != "chatml" else config.get("format", args.format)
    output_type = args.output_type if args.output_type != "json" else config.get("output_type", args.output_type)
    multimodal = args.multimodal or config.get("multimodal", False)
    dedupe = args.dedupe or config.get("dedupe", False)
    resume = args.resume or config.get("resume", False)
    show_stats = args.stats or config.get("stats", False)
    hub_repo = args.push_to_hub or config.get("push_to_hub")
    private = args.private or config.get("private", False)
    save_to_db = not args.no_db and config.get("save_to_db", True)
    quiet = args.quiet or config.get("quiet", False)
    verbose = args.verbose or config.get("verbose", False)
    keep_server = args.keep_server or config.get("keep_server", False)

    # Validate sources
    valid_sources = []
    for source in sources:
        if validate_source(source):
            valid_sources.append(source)

    if not valid_sources:
        print("Error: No valid sources provided.", file=sys.stderr)
        sys.exit(1)

    # Run generation
    success = run_generation(
        sources=valid_sources,
        output_path=output_path,
        gen_type=gen_type,
        num_pairs=num_pairs,
        max_chunks=max_chunks,
        curate=curate,
        curate_threshold=threshold,
        multimodal=multimodal,
        dataset_format=dataset_format,
        output_type=output_type,
        model=model,
        save_to_db=save_to_db,
        quiet=quiet,
        verbose=verbose,
        keep_server=keep_server,
        dedupe=dedupe,
        resume=resume,
        show_stats=show_stats,
        push_to_hub=hub_repo,
        private=private,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
