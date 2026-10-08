"""Run an eval set through retrieve -> generate -> score, and write reports."""

import json
import os
from datetime import datetime, timezone

from .generation import render_prompt
from .metrics import (
    answer_relevance,
    context_precision,
    context_recall,
    faithfulness,
)

METRICS = ["context_precision", "context_recall", "answer_relevance", "faithfulness"]


def evaluate(retriever, generator, prompt_path, eval_set, top_k=3):
    prompt_version = os.path.splitext(os.path.basename(prompt_path))[0]
    rows = []
    for item in eval_set:
        retrieved = retriever.retrieve(item["question"], top_k=top_k)
        prompt = render_prompt(prompt_path, item["question"], retrieved)
        answer = generator.generate(prompt, retrieved)
        retrieved_ids = [c["id"] for c in retrieved]
        scores = {
            "context_precision": context_precision(retrieved_ids, item["relevant_chunk_ids"]),
            "context_recall": context_recall(retrieved_ids, item["relevant_chunk_ids"]),
            "answer_relevance": answer_relevance(item["question"], answer),
            "faithfulness": faithfulness(answer, retrieved),
        }
        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "retrieved_ids": retrieved_ids,
                "answer": answer,
                "scores": scores,
            }
        )
    summary = {m: sum(r["scores"][m] for r in rows) / len(rows) for m in METRICS}
    return {
        "prompt_version": prompt_version,
        "top_k": top_k,
        "n_questions": len(rows),
        "summary": summary,
        "rows": rows,
    }


def write_reports(result, out_dir="reports"):
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    base = f"eval_{result['prompt_version']}_{stamp}"
    json_path = os.path.join(out_dir, base + ".json")
    md_path = os.path.join(out_dir, base + ".md")

    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    lines = [
        f"# RAG eval — `{result['prompt_version']}`",
        "",
        f"top_k={result['top_k']}, n={result['n_questions']}",
        "",
        "## Summary",
        "",
        "| metric | mean |",
        "|---|---|",
    ]
    for m in METRICS:
        lines.append(f"| {m} | {result['summary'][m]:.3f} |")
    lines += ["", "## Per-question", ""]
    lines.append("| id | question | " + " | ".join(METRICS) + " |")
    lines.append("|---|---|" + "---|" * len(METRICS))
    for r in result["rows"]:
        q = r["question"][:60].replace("|", "/")
        cells = " | ".join(f"{r['scores'][m]:.3f}" for m in METRICS)
        lines.append(f"| {r['id']} | {q} | {cells} |")

    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return md_path, json_path
