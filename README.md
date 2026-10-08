# RAG Evaluation Harness

A lightweight, dependency-light harness for evaluating Retrieval-Augmented Generation pipelines. Run a small eval set through any retriever + generator, score faithfulness, answer relevance, and context precision/recall, and compare prompt versions head-to-head.

Built to answer the interview question: *"How do you know your RAG system is any good?"*

## Why this exists

Most RAG demos stop at "it answered." This harness adds the missing loop: a versioned eval set, standard retrieval metrics, faithfulness and relevance scoring, and prompt versioning — so you can measure whether a change actually helped.

## Quickstart

```bash
pip install -r requirements.txt
python run_eval.py --prompt v1_baseline
python run_eval.py --prompt v2_grounded
```

Reports land in `reports/` as Markdown + JSON.

> Note: the bundled generator is an extractive mock that ignores the prompt,
> so both versions score identically out of the box. The version comparison
> becomes meaningful once you plug in a real LLM (see "Extending").

## How it works

1. **Retrieve** — `src/rag_eval/retrieval.py` ships a TF-IDF retriever over the toy corpus in `data/corpus.json`. Swap in your own retriever (anything with a `.retrieve(query, top_k)` method).
2. **Generate** — `src/rag_eval/generation.py` renders a versioned prompt from `prompts/` and calls a generator. The default is an extractive mock so the harness runs with zero API keys; plug in a real LLM via the documented interface.
3. **Score** — `src/rag_eval/metrics.py` computes per-question metrics and aggregates.
4. **Report** — `src/rag_eval/runner.py` writes a Markdown table + full JSON.

## Metrics

| Metric | What it measures |
|---|---|
| `context_precision` | Fraction of retrieved chunks that are relevant |
| `context_recall` | Fraction of relevant chunks that were retrieved |
| `answer_relevance` | Cosine similarity between question and answer (TF-IDF) |
| `faithfulness` | Share of answer content-words grounded in retrieved context |

The heuristic scorers run offline with no API keys. For production, swap in an LLM judge — see "Extending."

## Prompt versioning

Prompts live in `prompts/` as plain text with `{{question}}` and `{{context}}` slots. Every report is tagged with the prompt version, so `v1_baseline` vs `v2_grounded` is an apples-to-apples comparison.

## Extending

- **Real LLM**: implement `generate(prompt, contexts)` in `generation.py` (there's a commented example for an OpenAI-compatible client).
- **Real retriever**: implement `retrieve(query, top_k)` returning `[{"id", "text", "score"}]`.
- **LLM judge**: replace `faithfulness()` with a judge call; keep the heuristic as a cheap CI gate.
- **Embeddings**: swap TF-IDF for sentence-transformers in `answer_relevance` for a stronger signal.

## Project structure

```
rag-eval-harness/
├── data/               # toy corpus + labeled eval set (nutrition domain)
├── prompts/            # versioned prompt templates
├── src/rag_eval/       # retriever, generator, metrics, runner
├── run_eval.py         # CLI entry point
└── reports/            # generated (gitignored)
```

## Suggested resume bullet

> Built an offline RAG evaluation harness (Python, scikit-learn) scoring faithfulness, answer relevance, and context precision/recall across versioned prompts; used it to compare prompt strategies on a labeled eval set.
