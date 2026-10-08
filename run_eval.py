#!/usr/bin/env python3
"""CLI entry point.

Usage:
    python run_eval.py --prompt v1_baseline
    python run_eval.py --prompt v2_grounded --top-k 5
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from rag_eval.generation import ExtractiveGenerator
from rag_eval.retrieval import TfidfRetriever
from rag_eval.runner import evaluate, write_reports


def main():
    ap = argparse.ArgumentParser(description="Run the RAG eval harness.")
    ap.add_argument("--prompt", default="v1_baseline",
                    help="Prompt version in prompts/ (without .txt)")
    ap.add_argument("--top-k", type=int, default=3,
                    help="Number of chunks to retrieve per question")
    ap.add_argument("--data-dir", default="data",
                    help="Directory holding corpus.json and eval_set.json")
    args = ap.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base, args.data_dir, "corpus.json")) as f:
        corpus = json.load(f)
    with open(os.path.join(base, args.data_dir, "eval_set.json")) as f:
        eval_set = json.load(f)
    prompt_path = os.path.join(base, "prompts", args.prompt + ".txt")
    if not os.path.exists(prompt_path):
        ap.error(f"prompt not found: {prompt_path}")

    result = evaluate(
        TfidfRetriever(corpus),
        ExtractiveGenerator(),
        prompt_path,
        eval_set,
        top_k=args.top_k,
    )
    md_path, json_path = write_reports(result, out_dir=os.path.join(base, "reports"))

    print(f"prompt: {result['prompt_version']}   top_k={args.top_k}   n={result['n_questions']}")
    for m, v in result["summary"].items():
        print(f"  {m:20s} {v:.3f}")
    print(f"wrote {md_path}")
    print(f"      {json_path}")


if __name__ == "__main__":
    main()
