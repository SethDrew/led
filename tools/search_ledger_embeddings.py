#!/usr/bin/env python3
"""Semantic search over the LED project's research AND engineering ledgers.

Uses local embedding model (bge-small-en-v1.5 via mlx-embeddings) for
semantic similarity. Caches embeddings as numpy arrays next to ledger files;
rebuilds automatically when the ledger mtime changes.

Usage:
    python search_ledger_embeddings.py "query terms here"
    python search_ledger_embeddings.py --top 3 "query terms here"
"""

import argparse
import json
import os
import re
import sys
import time

import numpy as np
import yaml

# ── Paths ────────────────────────────────────────────────────────────

_BASE = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

LEDGER_PATHS = [
    os.path.join(_BASE, "audio-reactive", "research", "ledger.yaml"),
    os.path.join(_BASE, "engineering", "ledger.yaml"),
]

EMBEDDING_MODEL = "mlx-community/bge-small-en-v1.5-4bit"

MIN_SCORE = 0.30  # cosine similarity threshold (higher than TF-IDF since embeddings are denser)

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
    """Combine entry fields into a single string for embedding."""
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


# ── Embedding model ──────────────────────────────────────────────────

_model = None
_tokenizer = None


def _load_model():
    global _model, _tokenizer
    if _model is None:
        from mlx_embeddings.utils import load
        _model, _tokenizer = load(EMBEDDING_MODEL)
    return _model, _tokenizer


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of texts, returning (N, D) numpy array of unit vectors."""
    from mlx_embeddings.utils import generate
    import mlx.core as mx

    model, tokenizer = _load_model()
    output = generate(model, tokenizer, texts)
    emb = output.text_embeds if output.text_embeds is not None else output.pooler_output
    mx.eval(emb)
    arr = np.array(emb)
    # L2-normalize
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return arr / norms


# ── Cache management ─────────────────────────────────────────────────


def _cache_paths(ledger_path: str) -> tuple[str, str]:
    base = ledger_path + ".emb_cache"
    return base + ".npy", base + ".json"


def _get_combined_mtime() -> float:
    """Get the latest mtime across all ledger files."""
    return max(
        (os.path.getmtime(p) for p in LEDGER_PATHS if os.path.exists(p)),
        default=0.0,
    )


def _cache_path_combined() -> tuple[str, str]:
    """Single cache location for combined embeddings."""
    cache_dir = os.path.join(_BASE, ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    return (
        os.path.join(cache_dir, "ledger_embeddings.npy"),
        os.path.join(cache_dir, "ledger_metadata.json"),
    )


def get_cached_embeddings() -> tuple[list[tuple[str, dict]], np.ndarray] | None:
    """Load cached embeddings if they're still fresh."""
    npy_path, json_path = _cache_path_combined()
    if not os.path.exists(npy_path) or not os.path.exists(json_path):
        return None

    cache_mtime = min(os.path.getmtime(npy_path), os.path.getmtime(json_path))
    ledger_mtime = _get_combined_mtime()
    if cache_mtime < ledger_mtime:
        return None

    try:
        embeddings = np.load(npy_path)
        with open(json_path, "r") as f:
            metadata = json.load(f)
        # Reconstruct entries from metadata
        entries = []
        for item in metadata:
            entries.append((item["source"], item["entry"]))
        if len(entries) != embeddings.shape[0]:
            return None
        return entries, embeddings
    except (ValueError, KeyError, json.JSONDecodeError, OSError):
        return None


def build_and_cache_embeddings() -> tuple[list[tuple[str, dict]], np.ndarray]:
    """Build embeddings for all ledger entries and cache them."""
    entries = load_all_entries()
    if not entries:
        return [], np.empty((0, 0))

    texts = [entry_text(e) for _, e in entries]
    embeddings = embed_texts(texts)

    # Cache
    npy_path, json_path = _cache_path_combined()
    metadata = [{"source": src, "entry": entry} for src, entry in entries]
    try:
        np.save(npy_path, embeddings)
        with open(json_path, "w") as f:
            json.dump(metadata, f, default=str)
    except OSError:
        pass  # Cache write failure is non-fatal

    return entries, embeddings


def get_embeddings() -> tuple[list[tuple[str, dict]], np.ndarray]:
    """Get embeddings, using cache if available."""
    cached = get_cached_embeddings()
    if cached is not None:
        return cached
    return build_and_cache_embeddings()


# ── Reference doc matching (preserved from TF-IDF version) ───────────

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
    matches = []
    for doc_name, (pattern, description) in REFERENCE_DOCS.items():
        if re.search(pattern, query, re.IGNORECASE):
            matches.append((doc_name, description))
    return matches[:MAX_REFERENCE_DOCS]


# ── Search ───────────────────────────────────────────────────────────


def search(query: str, top_k: int = 5) -> str:
    """Search both ledgers and return formatted results."""
    lines = []

    entries, embeddings = get_embeddings()
    if not entries or embeddings.size == 0:
        return ""

    # Embed query
    query_emb = embed_texts([query])  # (1, D)

    # Cosine similarity (vectors are already normalized)
    scores = (embeddings @ query_emb.T).flatten()

    # Boost scores based on domain metadata.
    # Embedding cosine scores cluster tightly (typically 0.50-0.65 spread),
    # so boosts must be small relative to the score range.
    boosted = []
    for i, score in enumerate(scores):
        if score < MIN_SCORE:
            continue
        boost = 0.0
        _, entry = entries[i]
        # Research ledger fields
        warmth = entry.get("warmth", "")
        if warmth == "high":
            boost += 0.005
        status = entry.get("status", "")
        if status in ("validated", "integrated", "resonates"):
            boost += 0.005
        if entry.get("confidence", "") == "high":
            boost += 0.003
        # Engineering ledger fields
        severity = entry.get("severity", "")
        if severity == "high":
            boost += 0.005
        boosted.append((i, float(score) + boost))

    ranked = sorted(boosted, key=lambda x: x[1], reverse=True)[:top_k]

    if ranked:
        lines.append("## Relevant ledger entries:\n")
        for idx, score in ranked:
            source, entry = entries[idx]
            eid = entry.get("id", "unknown")
            status = entry.get("status", "?")

            # Format metadata depending on source
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

    # Reference docs
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
    parser = argparse.ArgumentParser(description="Search ledgers (embedding-based)")
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
