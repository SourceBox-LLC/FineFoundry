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
    model: str = "unsloth/Llama-3.2-3B-Instruct",
    save_to_db: bool = True,
    quiet: bool = False,
    verbose: bool = False,
    keep_server: bool = False,
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
        model: Model name to use for generation
        save_to_db: Save results to FineFoundry database
        quiet: Suppress all output except errors
        verbose: Show detailed debug output
        keep_server: Keep vLLM server running after generation (for batch runs)
    """
    start_time = time.time()

    log("🚀 Starting synthetic data generation", quiet)
    log(f"   Model: {model}", quiet)
    log(f"   Type: {gen_type}", quiet)
    log(f"   Sources: {len(sources)}", quiet)
    if verbose:
        debug(f"Output: {output_path}", verbose)
        debug(f"Num pairs per chunk: {num_pairs}", verbose)
        debug(f"Max chunks per source: {max_chunks}", verbose)
        debug(f"Curate: {curate} (threshold: {curate_threshold})", verbose)
        debug(f"Format: {dataset_format}", verbose)
        debug(f"Multimodal: {multimodal}", verbose)
        debug(f"Save to DB: {save_to_db}", verbose)
    log("", quiet)

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
            log(f"  🔄 Processing {len(chunks_to_process)} chunks...", quiet)

            # Create progress bar for chunks
            pbar = create_progress_bar(len(chunks_to_process), f"  {source_name[:20]}", quiet)

            for i, chunk_file in enumerate(chunks_to_process):
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

                if pbar:
                    pbar.update(1)

            if pbar:
                pbar.close()
            sources_processed += 1

        # Combine all datasets
        if all_conversations:
            log(f"\n📦 Combining {len(all_conversations)} conversations into final dataset...", quiet)

            combined_data = all_conversations

            if dataset_format.lower() == "standard":
                final_data = convert_to_standard(combined_data)
            else:
                final_data = convert_to_chatml(combined_data)

            total_pairs = len(final_data)
            pairs_output = dataset_format.lower() == "standard"

            # Write to output file
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(final_data, f, ensure_ascii=False, indent=2)

            log(f"💾 Saved to: {output_path}", quiet)

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

            # Print summary in quiet mode too
            if quiet:
                print(
                    json.dumps(
                        {
                            "success": True,
                            "output": output_path,
                            "pairs": total_pairs,
                            "sources": sources_processed,
                            "elapsed_seconds": round(total_elapsed, 2),
                        }
                    )
                )

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
        "--multimodal",
        action="store_true",
        help="Enable multimodal processing for images in documents",
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
    multimodal = args.multimodal or config.get("multimodal", False)
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
        model=model,
        save_to_db=save_to_db,
        quiet=quiet,
        verbose=verbose,
        keep_server=keep_server,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
