#!/usr/bin/env python3
"""
Pretty-print JSONL records for human readability.
Each input line should be a valid JSON object.
Writes multi-line, indented JSON objects, one record after another separated by a newline.

Usage:
  python scripts/pretty_jsonl.py --in path/to/input.jsonl --out path/to/output_pretty.jsonl
"""
import argparse
import json
import sys
from typing import Optional


def pretty_jsonl(in_path: str, out_path: str, indent: int = 2) -> None:
    count = 0
    with open(in_path, 'r', encoding='utf-8') as fi, open(out_path, 'w', encoding='utf-8') as fo:
        for line in fi:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[warn] Skipping invalid JSON on line {count+1}: {e}", file=sys.stderr)
                continue
            fo.write(json.dumps(obj, ensure_ascii=False, indent=indent) + '\n')
            count += 1
    print(f"[ok] Wrote {count} pretty-printed records to {out_path}")


def main(argv: Optional[list] = None) -> None:
    p = argparse.ArgumentParser(description="Pretty-print JSONL for easier reading")
    p.add_argument('--in', dest='inp', required=True, help='Input JSONL file path')
    p.add_argument('--out', dest='out', required=True, help='Output JSONL file path')
    p.add_argument('--indent', type=int, default=2, help='Indent size for pretty printing (default: 2)')
    args = p.parse_args(argv)

    pretty_jsonl(args.inp, args.out, indent=args.indent)


if __name__ == '__main__':
    main()
