#!/usr/bin/env python3
"""Semantic search over the LED project's research ledger.

Builds a TF-IDF index over ledger entries and returns top-K matches
for a query string. Caches the vectorizer and matrix in a pickle file
next to the ledger; rebuilds automatically when the ledger changes.

Usage:
    python search_ledger.py "query terms here"
    python search_ledger.py --top 3 "query terms here"
"""

import argparse
import os
import pickle
import sys

import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

LEDGER_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "audio-reactive",
    "research",
    "ledger.yaml",
)
LEDGER_PATH = os.path.normpath(LEDGER_PATH)
CACHE_PATH = LEDGER_PATH + ".search_cache.pkl"

MIN_SCORE = 0.05


def load_entries(ledger_path: str) -> list[dict]:
    """Parse ledger YAML and return list of entry dicts."""
    with open(ledger_path, "r") as f:
        data = yaml.safe_load(f)
    if not data or "entries" not in data:
        return []
    entries = data["entries"]
    if not isinstance(entries, list):
        return []
    return entries


def entry_text(entry: dict) -> str:
    """Combine entry fields into a single searchable string."""
    parts = []

    title = entry.get("title", "")
    if title:
        parts.append(str(title))

    summary = entry.get("summary", "")
    if summary:
        parts.append(str(summary))

    tags = entry.get("tags", [])
    if tags and isinstance(tags, list):
        parts.append(" ".join(str(t) for t in tags))

    notes = entry.get("notes", "")
    if notes:
        parts.append(str(notes))

    return " ".join(parts)


def build_index(entries: list[dict]):
    """Build TF-IDF vectorizer and matrix from entries."""
    corpus = [entry_text(e) for e in entries]
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_df=0.9,
        min_df=1,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix


def get_index(ledger_path: str):
    """Load or rebuild TF-IDF index with mtime-based cache invalidation."""
    entries = load_entries(ledger_path)
    if not entries:
        return entries, None, None

    ledger_mtime = os.path.getmtime(ledger_path)

    if os.path.exists(CACHE_PATH):
        try:
            cache_mtime = os.path.getmtime(CACHE_PATH)
            if cache_mtime >= ledger_mtime:
                with open(CACHE_PATH, "rb") as f:
                    cached = pickle.load(f)
                if cached.get("entry_count") == len(entries):
                    return entries, cached["vectorizer"], cached["matrix"]
        except (pickle.UnpicklingError, KeyError, EOFError, OSError):
            pass

    vectorizer, matrix = build_index(entries)

    try:
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(
                {
                    "vectorizer": vectorizer,
                    "matrix": matrix,
                    "entry_count": len(entries),
                },
                f,
            )
    except OSError:
        pass  # Cache write failure is non-fatal

    return entries, vectorizer, matrix


def search(query: str, top_k: int = 5) -> str:
    """Search the ledger and return formatted results."""
    entries, vectorizer, matrix = get_index(LEDGER_PATH)
    if not entries or vectorizer is None:
        return ""

    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, matrix).flatten()

    # Boost scores based on domain metadata (warmth, status, confidence)
    boosted = []
    for i, score in enumerate(scores):
        if score < MIN_SCORE:
            continue
        boost = 0.0
        entry = entries[i]
        warmth = entry.get("warmth", "")
        if warmth == "high":
            boost += 0.03
        status = entry.get("status", "")
        if status in ("validated", "integrated", "resonates"):
            boost += 0.02
        if entry.get("confidence", "") == "high":
            boost += 0.01
        boosted.append((i, score + boost))

    ranked = sorted(boosted, key=lambda x: x[1], reverse=True)[:top_k]

    if not ranked:
        return ""

    lines = ["## Relevant research ledger entries:\n"]
    for idx, score in ranked:
        entry = entries[idx]
        eid = entry.get("id", "unknown")
        status = entry.get("status", "?")
        warmth = entry.get("warmth", "?")
        confidence = entry.get("confidence", "?")

        summary = entry.get("summary", "").strip()
        if not summary:
            summary = entry.get("title", "(no summary)")

        relates = entry.get("relates_to", [])
        if relates and isinstance(relates, list):
            relates_str = ", ".join(str(r) for r in relates)
        else:
            relates_str = ""

        lines.append(
            f"**{eid}** [{status}, warmth:{warmth}, confidence:{confidence}]"
        )
        lines.append(f"{summary}")
        if relates_str:
            lines.append(f"Related: [{relates_str}]")
        lines.append("")

    return "\n".join(lines).rstrip()


def main():
    parser = argparse.ArgumentParser(description="Search the research ledger")
    parser.add_argument("query", help="Search query string")
    parser.add_argument(
        "--top", type=int, default=5, help="Number of results (default: 5)"
    )
    args = parser.parse_args()

    result = search(args.query, top_k=args.top)
    if result:
        print(result)


if __name__ == "__main__":
    main()
