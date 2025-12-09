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
from typing import List, Optional

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


def log(msg: str, quiet: bool = False) -> None:
    """Print log message unless quiet mode."""
    if not quiet:
        print(msg)


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


def convert_to_ft_format(json_file: Path, config_path: Path, quiet: bool) -> Optional[Path]:
    """Convert to fine-tuning format."""
    import subprocess

    cmd = [
        "synthetic-data-kit",
        "-c",
        str(config_path),
        "convert",
        str(json_file),
        "--format",
        "ft",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        if not quiet:
            print(f"    ⚠️ Conversion warning: {result.stderr[:100]}")
        return None

    ft_path = Path(f"data/final/{json_file.stem}_ft.json")
    if ft_path.exists():
        return ft_path

    for f in Path("data/final").glob(f"{json_file.stem}*.json"):
        return f

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
) -> bool:
    """Run the full synthetic data generation pipeline."""
    start_time = time.time()

    log("🚀 Starting synthetic data generation", quiet)
    log(f"   Model: {model}", quiet)
    log(f"   Type: {gen_type}", quiet)
    log(f"   Sources: {len(sources)}", quiet)
    log("", quiet)

    # Clear and create output directories
    for folder in ["data/output", "data/generated", "data/curated", "data/final"]:
        if Path(folder).exists():
            shutil.rmtree(folder)
        Path(folder).mkdir(parents=True, exist_ok=True)

    config_path = create_config_file()

    # Load model
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

    all_ft_files = []
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

            for i, chunk_file in enumerate(chunks_to_process):
                chunk_start = time.time()
                log(f"\n  [{i + 1}/{len(chunks_to_process)}] Generating {gen_type}...", quiet)

                # Generate content
                gen_file = generate_content(chunk_file, config_path, gen_type, num_pairs, quiet)
                if not gen_file:
                    log("    ❌ Generation failed for this chunk", quiet)
                    continue

                # Optionally curate
                if curate:
                    gen_file = curate_content(gen_file, config_path, curate_threshold, quiet)

                # Convert to fine-tuning format
                ft_file = convert_to_ft_format(gen_file, config_path, quiet)
                if ft_file:
                    all_ft_files.append(ft_file)
                    total_pairs += num_pairs
                    chunk_elapsed = time.time() - chunk_start
                    log(f"  ✅ Generated: {ft_file.name} ({format_time(chunk_elapsed)})", quiet)

            sources_processed += 1

        # Combine all datasets
        if all_ft_files:
            log(f"\n📦 Combining {len(all_ft_files)} files into final dataset...", quiet)

            import pandas as pd

            conversations = pd.concat([pd.read_json(f) for f in all_ft_files]).reset_index(drop=True)
            combined_data = conversations.to_dict(orient="records")

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

    # Required arguments
    parser.add_argument(
        "--source",
        "-s",
        action="append",
        required=True,
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

    args = parser.parse_args()

    # Validate sources
    valid_sources = []
    for source in args.source:
        if validate_source(source):
            valid_sources.append(source)

    if not valid_sources:
        print("Error: No valid sources provided.", file=sys.stderr)
        sys.exit(1)

    # Run generation
    success = run_generation(
        sources=valid_sources,
        output_path=args.output,
        gen_type=args.type,
        num_pairs=args.num_pairs,
        max_chunks=args.max_chunks,
        curate=args.curate,
        curate_threshold=args.threshold,
        multimodal=args.multimodal,
        dataset_format=args.format,
        model=args.model,
        save_to_db=not args.no_db,
        quiet=args.quiet,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
