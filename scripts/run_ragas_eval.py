#!/usr/bin/env python3
"""
Offline RAG evaluation using RAGAS (faithfulness + answer relevancy).

Runs the real agent (same path as production), then scores answers against
retrieved contexts. Requires OPENAI_API_KEY and a built FAISS index.

Usage (from Enterprise Knowledge Transfer Agent/):
  pip install ragas datasets
  PYTHONPATH=src .venv/bin/python scripts/run_ragas_eval.py
  PYTHONPATH=src .venv/bin/python scripts/run_ragas_eval.py --questions data/eval_questions.json
  PYTHONPATH=src .venv/bin/python scripts/run_ragas_eval.py --output data/ragas_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

# LangGraph + structured outputs can trigger pydantic serializer UserWarnings during checkpointing.
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"pydantic\.main",
)
# RAGAS still expects LangChain-style embed_query(); wrapper is deprecated but required for metrics.
warnings.filterwarnings(
    "ignore",
    message=".*LangchainEmbeddingsWrapper is deprecated.*",
    category=DeprecationWarning,
)

# Project root = parent of scripts/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def _load_env() -> None:
    env_file = ROOT / ".env"
    if not env_file.is_file():
        return
    import os

    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        key = k.strip()
        val = v.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def _contexts_from_docs(docs: list) -> list[str]:
    out: list[str] = []
    for d in docs:
        text = getattr(d, "page_content", None) or ""
        if text.strip():
            out.append(text)
    return out if out else ["(no chunks retrieved)"]


def main() -> int:
    _load_env()

    parser = argparse.ArgumentParser(description="Run RAGAS eval on the knowledge transfer agent.")
    parser.add_argument(
        "--questions",
        type=Path,
        default=ROOT / "data" / "eval_questions.json",
        help="JSON array of objects with a 'question' field.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write RAGAS scores as JSON (e.g. data/ragas_results.json).",
    )
    args = parser.parse_args()

    try:
        from datasets import Dataset
        from openai import OpenAI
        from ragas import evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms.base import llm_factory
    except ImportError as e:
        print("Missing dependency. Install with: pip install ragas datasets", file=sys.stderr)
        print(e, file=sys.stderr)
        return 1

    try:
        from ragas.metrics._faithfulness import faithfulness as faithfulness_metric
        from ragas.metrics._answer_relevance import answer_relevancy as answer_relevancy_metric
    except ImportError:
        from ragas.metrics import faithfulness as faithfulness_metric
        from ragas.metrics import answer_relevancy as answer_relevancy_metric

    from knowledge_transfer_agent.config import get_settings
    from knowledge_transfer_agent.retrieval.embeddings import get_embeddings
    from knowledge_transfer_agent.retrieval.vector_store import get_vector_store
    from knowledge_transfer_agent.services.agent_service import AgentService

    settings = get_settings()
    path = args.questions
    if not path.is_file():
        print(f"Questions file not found: {path}", file=sys.stderr)
        return 1

    items = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(items, list) or not items:
        print("eval_questions.json must be a non-empty JSON array.", file=sys.stderr)
        return 1

    client = OpenAI(api_key=settings.openai_api_key)
    ragas_llm = llm_factory(settings.openai_model, client=client, temperature=0)
    emb = LangchainEmbeddingsWrapper(get_embeddings())

    user_inputs: list[str] = []
    responses: list[str] = []
    all_contexts: list[list[str]] = []

    try:
        vs = get_vector_store()
    except FileNotFoundError as e:
        print(
            "FAISS index not found. Run ingestion first (build VECTOR_STORE_PATH).",
            file=sys.stderr,
        )
        print(e, file=sys.stderr)
        return 1

    service = AgentService(vector_store=vs)

    for item in items:
        q = (item.get("question") or "").strip()
        if not q:
            continue
        result = service.ask(q, thread_id="ragas-eval")
        user_inputs.append(q)
        responses.append(result.get("answer") or "")
        all_contexts.append(_contexts_from_docs(result.get("retrieved_docs") or []))

    if not user_inputs:
        print("No valid questions in file.", file=sys.stderr)
        return 1

    ds = Dataset.from_dict(
        {
            "user_input": user_inputs,
            "response": responses,
            "retrieved_contexts": all_contexts,
        }
    )

    result = evaluate(
        ds,
        metrics=[faithfulness_metric, answer_relevancy_metric],
        llm=ragas_llm,
        embeddings=emb,
        raise_exceptions=False,
    )
    print(result)
    if args.output is not None:
        out_path = args.output if args.output.is_absolute() else ROOT / args.output
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            from ragas.dataset_schema import EvaluationResult

            if isinstance(result, EvaluationResult):
                serializable = {
                    "aggregate": {k: float(v) for k, v in result._repr_dict.items()},
                    "per_question": result.scores,
                }
            else:
                serializable = {"result": str(result)}
        except Exception:
            serializable = {"result": str(result)}
        out_path.write_text(json.dumps(serializable, indent=2, default=str), encoding="utf-8")
        print(f"Wrote scores to {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
