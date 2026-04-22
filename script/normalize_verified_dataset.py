#!/usr/bin/env python3
"""Normalize cab_verified_v2.jsonl into CAB-consumable JSONL.

This fixes fields that were serialized as Python-literal strings instead of
native JSON arrays/objects, such as:
  - comments
  - satisfaction_conditions
  - labels
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


DEFAULT_INPUT = "dataset/cab_verified_v2.jsonl"
DEFAULT_OUTPUT = "dataset/cab_verified_v2_fixed.jsonl"

FIELDS_TO_FIX = {
    "comments": list,
    "satisfaction_conditions": list,
    "labels": list,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input JSONL path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output JSONL path")
    return parser.parse_args()


def maybe_parse_literal(value):
    if not isinstance(value, str):
        return value, False

    text = value.strip()
    if not text:
        return value, False

    try:
        return ast.literal_eval(text), True
    except (SyntaxError, ValueError):
        return value, False


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    fixed_counts = {field: 0 for field in FIELDS_TO_FIX}
    type_mismatches = {field: 0 for field in FIELDS_TO_FIX}
    total = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line_no, line in enumerate(fin, start=1):
            total += 1
            obj = json.loads(line)

            for field, expected_type in FIELDS_TO_FIX.items():
                original = obj.get(field)
                parsed, changed = maybe_parse_literal(original)
                if changed:
                    obj[field] = parsed
                    fixed_counts[field] += 1

                if not isinstance(obj.get(field), expected_type):
                    type_mismatches[field] += 1
                    raise TypeError(
                        f"Line {line_no}: field {field!r} has type "
                        f"{type(obj.get(field)).__name__}, expected {expected_type.__name__}"
                    )

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Normalized {total} records")
    print(f"Output: {output_path}")
    for field in FIELDS_TO_FIX:
        print(
            f"{field}: fixed={fixed_counts[field]}, mismatches={type_mismatches[field]}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
