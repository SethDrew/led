#!/usr/bin/env python3
"""TF-IDF search over the LED project's research AND engineering ledgers.

Builds a TF-IDF index over ledger entries and returns top-K matches
for a query string. Caches the vectorizer and matrix in a pickle file;
rebuilds automatically when any ledger mtime changes.

Usage (standalone):
    python search_ledger_tfidf.py "query terms here"
    python search_ledger_tfidf.py --top 5 "query terms here"

Also importable: search_raw() returns (entry_id, source, entry_dict, score) tuples.
"""

import argparse
import os
import pickle
import re
import sys

import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Paths ────────────────────────────────────────────────────────────

_BASE = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

LEDGER_PATHS = [
    os.path.join(_BASE, "audio-reactive", "research", "ledger.yaml"),
    os.path.join(_BASE, "engineering", "ledger.yaml"),
]

CACHE_DIR = os.path.join(_BASE, ".cache")
CACHE_PATH = os.path.join(CACHE_DIR, "ledger_tfidf_cache.pkl")

MIN_SCORE = 0.05

# ── Ledger loading ───────────────────────────────────────────────────


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


def load_all_entries() -> list[tuple[str, dict]]:
    """Load entries from all ledgers. Returns (source_label, entry) pairs."""
    all_entries = []
    for path in LEDGER_PATHS:
        if not os.path.exists(path):
            continue
        label = "research" if "research" in path else "engineering"
        for entry in load_entries(path):
            all_entries.append((label, entry))
    return all_entries


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


# ── TF-IDF index ────────────────────────────────────────────────────


def _get_combined_mtime() -> float:
    """Get the latest mtime across all ledger files."""
    return max(
        (os.path.getmtime(p) for p in LEDGER_PATHS if os.path.exists(p)),
        default=0.0,
    )


def build_index(entries: list[tuple[str, dict]]):
    """Build TF-IDF vectorizer and matrix from entries."""
    corpus = [entry_text(e) for _, e in entries]
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_df=0.9,
        min_df=1,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix


def get_index():
    """Load or rebuild TF-IDF index with mtime-based cache invalidation."""
    entries = load_all_entries()
    if not entries:
        return entries, None, None

    ledger_mtime = _get_combined_mtime()
    os.makedirs(CACHE_DIR, exist_ok=True)

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


# ── Search ───────────────────────────────────────────────────────────


def search_raw(query: str, top_k: int = 5) -> list[tuple[str, str, dict, float]]:
    """Return raw search results as (entry_id, source, entry_dict, score) tuples.

    This is the interface used by the combined search script.
    """
    entries, vectorizer, matrix = get_index()
    if not entries or vectorizer is None:
        return []

    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, matrix).flatten()

    boosted = []
    for i, score in enumerate(scores):
        if score < MIN_SCORE:
            continue
        boost = 0.0
        _, entry = entries[i]
        # Research ledger fields
        warmth = entry.get("warmth", "")
        if warmth == "high":
            boost += 0.03
        status = entry.get("status", "")
        if status in ("validated", "integrated", "resonates"):
            boost += 0.02
        if entry.get("confidence", "") == "high":
            boost += 0.01
        # Engineering ledger fields
        severity = entry.get("severity", "")
        if severity == "high":
            boost += 0.03
        boosted.append((i, score + boost))

    ranked = sorted(boosted, key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for idx, score in ranked:
        source, entry = entries[idx]
        eid = entry.get("id", "unknown")
        results.append((eid, source, entry, float(score)))
    return results


def format_results(results: list[tuple[str, str, dict, float]]) -> list[str]:
    """Format raw results into output lines (no header)."""
    lines = []
    for eid, source, entry, score in results:
        status = entry.get("status", "?")

        if source == "research":
            warmth = entry.get("warmth", "?")
            confidence = entry.get("confidence", "?")
            meta = f"[{source}, {status}, warmth:{warmth}, confidence:{confidence}]"
        else:
            severity = entry.get("severity", "?")
            scope = entry.get("scope", "?")
            meta = f"[{source}, {status}, severity:{severity}, scope:{scope}]"

        summary = entry.get("summary", "").strip()
        if not summary:
            summary = entry.get("title", "(no summary)")

        relates = entry.get("relates_to", [])
        if relates and isinstance(relates, list):
            relates_str = ", ".join(str(r) for r in relates)
        else:
            relates_str = ""

        lines.append(f"**{eid}** {meta}")
        lines.append(f"{summary}")
        if relates_str:
            lines.append(f"Related: [{relates_str}]")
        lines.append("")
    return lines


def search(query: str, top_k: int = 5) -> str:
    """Search both ledgers and return formatted results."""
    results = search_raw(query, top_k=top_k)
    lines = []
    if results:
        lines.append("## Relevant ledger entries:\n")
        lines.extend(format_results(results))
    return "\n".join(lines).rstrip()


def main():
    parser = argparse.ArgumentParser(description="Search ledgers (TF-IDF)")
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
