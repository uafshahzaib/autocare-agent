"""
Retrieval-quality evaluation for the RAG component, in the spirit of the
evaluation methodology in Lewis et al. 2020 ("Retrieval-Augmented Generation
for Knowledge-Intensive NLP Tasks"): before trusting the generator, check
that retrieval itself is finding the right source document.

Metric: hit-rate@k — for each labeled question, does the correct source
document appear among the top-k retrieved chunks?

Usage:
    python -m eval.evaluate            # k=3 (default, matches app/tools.py)
    python -m eval.evaluate --k 1      # stricter: top-1 only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.ingest import load_vector_store

EVAL_FILE = Path(__file__).parent / "qa_eval.jsonl"


def load_eval_set() -> list[dict]:
    with open(EVAL_FILE) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_eval(k: int = 3) -> None:
    vector_store = load_vector_store()
    eval_set = load_eval_set()

    hits = 0
    print(f"Running retrieval eval (k={k}) on {len(eval_set)} labeled questions...\n")
    for item in eval_set:
        question = item["question"]
        expected_source = item["relevant_source"]

        results = vector_store.similarity_search(question, k=k)
        retrieved_sources = {
            Path(doc.metadata.get("source", "")).name for doc in results
        }

        hit = expected_source in retrieved_sources
        hits += int(hit)

        status = "HIT " if hit else "MISS"
        print(f"[{status}] '{question}' -> expected {expected_source}, got {sorted(retrieved_sources)}")

    hit_rate = hits / len(eval_set)
    print(f"\nRetrieval hit-rate@{k}: {hits}/{len(eval_set)} = {hit_rate:.0%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()
    run_eval(k=args.k)
