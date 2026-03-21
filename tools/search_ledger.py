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
import re
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


# Keyword → arch-impl reference doc mapping.
# Each value is (regex_pattern, brief_description).
REFERENCE_DOCS = {
    "COLOR_ENGINEERING.md": (
        r"colou?r|rainbow|palette|gamma|hue|saturation|chroma|oklch|brightness|rgb",
        "color pipeline, gamma, OKLCH, chroma",
    ),
    "SIGNAL_NORMALIZATION.md": (
        r"normali[zs]|scaling|ema|time.?constant|smoothing|agc|envelope",
        "EMA, scaling, AGC, envelope following",
    ),
    "MATHY_EFFECTS_CATALOG.md": (
        r"effect.?design|generative|math.?effect|pattern|perlin|simplex|wave|oscillat",
        "generative patterns, Perlin noise, wave math",
    ),
    "NATURE_TOPOLOGY_EFFECTS_CATALOG.md": (
        r"topolog|tree.?form|diamond|branch|sculpture|canopy|radial",
        "topology-specific effects, tree/diamond forms",
    ),
    "VJ_AUDIO_VISUAL_MAPPING.md": (
        r"vj|audio.?visual|mapping.?conven|lighting.?design|ld\b",
        "VJ conventions, audio-visual mapping principles",
    ),
    "SPECTROGRAM_BASED_COLOR.md": (
        r"spectro|mel.?to.?color|frequency.?color|spectral.?color",
        "mel-to-color strategies, spectral color mapping",
    ),
    "ENTITY_INTERACTIONS.md": (
        r"entit|collision|flock|swarm|multi.?agent|particle.?interact",
        "multi-entity behavior, collision, flocking",
    ),
    "INPUT_ROLE_MATRIX.md": (
        r"input.?role|composition|layer.?role|foreground|background|midground",
        "input roles, layer composition",
    ),
    "SHOW_AUTOMATION.md": (
        r"show.?auto|scene.?change|phrase.?detect|arc|set.?list|transition",
        "show automation, scene transitions, phrase detection",
    ),
    "AUDIO_FEATURES.md": (
        r"audio.?feature|onset|spectral.?flux|centroid|rms|zcr|mfcc",
        "audio features: onset, flux, centroid, RMS, MFCC",
    ),
    "AUDIO_ANALYSIS_ALGORITHMS.md": (
        r"algorithm|fft|stft|hpss|decompos|harmonic|percussive",
        "FFT, STFT, HPSS, harmonic/percussive separation",
    ),
    "PER_BAND_NORMALIZATION.md": (
        r"per.?band|band.?normal|frequency.?band|sub.?bass|bass|mid|treble",
        "per-band normalization, frequency band processing",
    ),
    "AUDIO_VISUAL_MAPPING_PATTERNS.md": (
        r"mapping.?pattern|reactive.?pattern|pulse.?map|beat.?map",
        "reactive mapping patterns, beat/pulse mapping",
    ),
}

MAX_REFERENCE_DOCS = 3


def match_reference_docs(query: str) -> list[tuple[str, str]]:
    """Match query against keyword patterns to find relevant arch-impl docs.

    Returns list of (doc_name, description) tuples, capped at MAX_REFERENCE_DOCS.
    """
    matches = []
    for doc_name, (pattern, description) in REFERENCE_DOCS.items():
        if re.search(pattern, query, re.IGNORECASE):
            matches.append((doc_name, description))
    return matches[:MAX_REFERENCE_DOCS]


def search(query: str, top_k: int = 5) -> str:
    """Search the ledger and return formatted results."""
    lines = []

    # TF-IDF ledger search
    entries, vectorizer, matrix = get_index(LEDGER_PATH)
    if entries and vectorizer is not None:
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

        if ranked:
            lines.append("## Relevant research ledger entries:\n")
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

    # Keyword-matched reference docs
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
