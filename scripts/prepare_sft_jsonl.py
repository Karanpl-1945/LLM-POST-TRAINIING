#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Convert prompt/response JSONL into conversational SFT JSONL")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--prompt-field", default="prompt")
    p.add_argument("--response-field", default="response")
    args = p.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with Path(args.input).open("r", encoding="utf-8") as fin, out.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = str(row[args.prompt_field]).strip()
            response = str(row[args.response_field]).strip()
            if not prompt or not response:
                continue
            fout.write(json.dumps({"messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]}, ensure_ascii=False) + "\n")
            written += 1
    print(f"wrote {written:,} records to {out}")


if __name__ == "__main__":
    main()
