#!/usr/bin/env python3
"""Combined TF-IDF + embedding search over the LED project's ledgers.

Runs both search methods and returns the union of results (deduplicated by
entry ID), up to 10 entries total. TF-IDF is fast and good at keyword matches;
embeddings catch semantic/conceptual similarity that keywords miss.

Usage:
    python search_ledger_combined.py "query terms here"
    python search_ledger_combined.py --top 10 "query terms here"
"""

import argparse
import sys
import os

# Ensure tools/ is importable
sys.path.insert(0, os.path.dirname(__file__))

from search_ledger_tfidf import search_raw as tfidf_search_raw
from search_ledger_embeddings import (
    search_raw as embedding_search_raw,
    match_reference_docs,
    format_results,
)


def combined_search(query: str, top_k: int = 10) -> str:
    """Run both TF-IDF and embedding searches, return union up to top_k."""
    # Each searcher returns up to top_k results independently.
    # We ask each for top_k so we have enough candidates before dedup.
    tfidf_results = tfidf_search_raw(query, top_k=top_k)
    embedding_results = embedding_search_raw(query, top_k=top_k)

    # Build union by entry ID, keeping the higher-ranked position.
    # Strategy: interleave from both lists (embedding first since it's
    # generally more semantically relevant), skip duplicates.
    seen_ids = set()
    merged = []

    # Interleave: take one from embeddings, one from TF-IDF, repeat.
    # This ensures both methods contribute to the final list rather than
    # one dominating.
    ei, ti = 0, 0
    while len(merged) < top_k and (ei < len(embedding_results) or ti < len(tfidf_results)):
        # Take from embeddings
        while ei < len(embedding_results) and len(merged) < top_k:
            eid = embedding_results[ei][0]
            if eid not in seen_ids:
                seen_ids.add(eid)
                merged.append(embedding_results[ei])
                ei += 1
                break
            ei += 1

        # Take from TF-IDF
        while ti < len(tfidf_results) and len(merged) < top_k:
            eid = tfidf_results[ti][0]
            if eid not in seen_ids:
                seen_ids.add(eid)
                merged.append(tfidf_results[ti])
                ti += 1
                break
            ti += 1

    # If we haven't filled top_k yet, drain remaining from either list
    for result in embedding_results[ei:]:
        if len(merged) >= top_k:
            break
        if result[0] not in seen_ids:
            seen_ids.add(result[0])
            merged.append(result)

    for result in tfidf_results[ti:]:
        if len(merged) >= top_k:
            break
        if result[0] not in seen_ids:
            seen_ids.add(result[0])
            merged.append(result)

    # Format output
    lines = []
    if merged:
        lines.append("## Relevant ledger entries:\n")
        lines.extend(format_results(merged))

    # Reference docs (keyword-based, from embeddings module)
    ref_matches = match_reference_docs(query)
    if ref_matches:
        if lines:
            lines.append("")
        lines.append("## Relevant reference documents:")
        lines.append("Read these arch-impl docs if working on related topics:")
        for doc_name, description in ref_matches:
            lines.append(f"- {doc_name} ({description})")

    return "\n".join(lines).rstrip()


def main():
    parser = argparse.ArgumentParser(
        description="Combined TF-IDF + embedding ledger search"
    )
    parser.add_argument("query", help="Search query string")
    parser.add_argument(
        "--top", type=int, default=10, help="Max results (default: 10)"
    )
    args = parser.parse_args()

    result = combined_search(args.query, top_k=args.top)
    if result:
        print(result)


if __name__ == "__main__":
    main()
