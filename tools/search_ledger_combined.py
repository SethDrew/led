#!/usr/bin/env python3
"""Combined TF-IDF + embedding search over the LED project's ledgers.

Runs both search methods and returns the union of results (deduplicated by
entry ID), up to 10 entries total. TF-IDF is fast and good at keyword matches;
embeddings catch semantic/conceptual similarity that keywords miss.

Two graphify-inspired layers on top:
- Lexical tier: entries whose ID appears verbatim in the query, or whose tags
  exactly match query tokens (including joined bigrams, so "brown out" hits a
  "brownout" tag), are promoted above the semantic ranking. Fires on >=2
  matched tags or a single near-unique tag (document frequency <= 2).
- Expansion: relates_to links of the top results are appended as labeled
  neighbor entries (summary only), so the ledger graph contributes to recall.

Usage:
    python search_ledger_combined.py "query terms here"
    python search_ledger_combined.py --top 10 "query terms here"
"""

import argparse
import re
import sys
import os

# Ensure tools/ is importable
sys.path.insert(0, os.path.dirname(__file__))

from search_ledger_tfidf import search_raw as tfidf_search_raw
from search_ledger_embeddings import (
    search_raw as embedding_search_raw,
    load_all_entries,
    match_reference_docs,
    format_results,
)

EXPANSION_CAP = 4
LEXICAL_CAP = 3


def _entries_by_id() -> dict[str, tuple[str, dict]]:
    return {
        str(entry["id"]): (source, entry)
        for source, entry in load_all_entries()
        if entry.get("id")
    }


def _tag_df(entries_by_id: dict) -> dict[str, int]:
    df: dict[str, int] = {}
    for _, entry in entries_by_id.values():
        for tag in set(str(t).lower() for t in (entry.get("tags") or [])):
            df[tag] = df.get(tag, 0) + 1
    return df


def lexical_tier(
    query: str, entries_by_id: dict, tag_df: dict
) -> list[tuple[str, str, dict, float]]:
    q = query.lower()
    words = re.findall(r"[a-z0-9]+", q)
    tokens = {w for w in words if len(w) >= 4}
    tokens |= {a + b for a, b in zip(words, words[1:])}
    tokens |= {a + "-" + b for a, b in zip(words, words[1:])}

    hits = []
    for eid, (source, entry) in entries_by_id.items():
        if eid.lower() in q:
            hits.append((eid, source, entry, 100.0))
            continue
        matched = [
            tag
            for tag in set(str(t).lower() for t in (entry.get("tags") or []))
            if tag in tokens
        ]
        rare = [t for t in matched if tag_df.get(t, 1) <= 2]
        if len(matched) >= 2 or rare:
            score = sum(1.0 / tag_df.get(t, 1) for t in matched)
            hits.append((eid, source, entry, score))
    hits.sort(key=lambda h: h[3], reverse=True)
    return hits[:LEXICAL_CAP]


def expand_relates_to(
    merged: list, entries_by_id: dict
) -> list[tuple[str, str, dict, str]]:
    seen = {r[0] for r in merged}
    neighbors = []
    for eid, _source, entry, _score in merged[:5]:
        for rel in entry.get("relates_to") or []:
            rel = str(rel)
            if rel in seen or rel not in entries_by_id:
                continue
            seen.add(rel)
            nsource, nentry = entries_by_id[rel]
            neighbors.append((rel, nsource, nentry, eid))
            if len(neighbors) >= EXPANSION_CAP:
                return neighbors
    return neighbors


def combined_search(query: str, top_k: int = 10) -> str:
    """Run both TF-IDF and embedding searches, return union up to top_k."""
    # Each searcher returns up to top_k results independently.
    # We ask each for top_k so we have enough candidates before dedup.
    tfidf_results = tfidf_search_raw(query, top_k=top_k)
    embedding_results = embedding_search_raw(query, top_k=top_k)

    entries_by_id = _entries_by_id()
    tag_df = _tag_df(entries_by_id)

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

    lex_hits = lexical_tier(query, entries_by_id, tag_df)
    if lex_hits:
        lex_ids = {h[0] for h in lex_hits}
        merged = list(lex_hits) + [r for r in merged if r[0] not in lex_ids]
        merged = merged[:top_k]

    neighbors = expand_relates_to(merged, entries_by_id)

    # Format output
    lines = []
    if merged:
        lines.append("## Relevant ledger entries:\n")
        lines.extend(format_results(merged))
    if neighbors:
        lines.append("### Linked entries (via relates_to):\n")
        for eid, _nsource, nentry, via in neighbors:
            summary = (nentry.get("summary") or nentry.get("title") or "").strip()
            lines.append(f"**{eid}** [linked from {via}]")
            lines.append(summary)
            lines.append("")

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
