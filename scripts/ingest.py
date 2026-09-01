#!/usr/bin/env python3
"""
CLI script to run the ingestion pipeline.
Supports TXT and PDF via document loader.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from knowledge_transfer_agent.ingestion.embedding_pipeline import (
    add_to_existing_index,
    run_embedding_pipeline,
)
from knowledge_transfer_agent.ingestion.pipeline import IngestionPipeline
from knowledge_transfer_agent.logging_config import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest TXT/PDF documents into the knowledge base (load → chunk → embed → FAISS)"
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="File or directory paths to ingest (e.g., ./docs ./runbooks)",
    )
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Don't persist the FAISS index to disk",
    )
    parser.add_argument(
        "--add",
        action="store_true",
        help="Add to existing index instead of creating new one",
    )
    parser.add_argument(
        "--full-pipeline",
        action="store_true",
        help="Use full pipeline (Confluence, GitHub, files) instead of embedding-only",
    )
    parser.add_argument(
        "--replace-index",
        action="store_true",
        help="Rebuild FAISS from this run only (no merge; drops old vectors missing page metadata)",
    )
    args = parser.parse_args()

    setup_logging()

    if args.full_pipeline or not args.paths:
        # Full pipeline: ingestors + file paths
        pipeline = IngestionPipeline()
        results = pipeline.run(
            additional_paths=args.paths or None,
            persist=not args.no_persist,
            replace_index=args.replace_index,
        )
        for source, result in results.items():
            print(
                f"{source}: processed={result.documents_processed}, "
                f"failed={result.documents_failed}"
            )
            for err in result.errors[:3]:
                print(f"  Error: {err}")
        success = any(r.success for r in results.values())
    else:
        # Embedding pipeline only: load → chunk → embed → save
        if args.add:
            stats = add_to_existing_index(
                args.paths,
                recursive=True,
            )
        else:
            stats = run_embedding_pipeline(
                args.paths,
                recursive=True,
                persist=not args.no_persist,
            )
        print(
            f"Documents: {stats.get('documents_loaded', 0)}, "
            f"Chunks: {stats.get('chunks_created', stats.get('chunks_added', 0))}"
        )
        success = stats.get("documents_loaded", 0) > 0 or stats.get("chunks_added", 0) > 0

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
