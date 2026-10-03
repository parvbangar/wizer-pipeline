"""
enrichment/steps/embedding.py
══════════════════════════════
Multilingual sentence embeddings for story clustering.

WHAT THIS STEP PRODUCES:
  One L2-normalised 768-dim vector per article. Two articles about the same
  real-world event — in the same language or not — land close together
  (cosine similarity ≥ ~0.8); unrelated stories land far apart.
  The vector is handed to the SQL function wizer_assign_cluster(), which does
  the actual clustering (see enrichment/clustering.py and docs/CLUSTERING.md).

WHAT TEXT GETS EMBEDDED (build_embedding_text):
  headline + the first ~300 chars of the RSS description; when the description
  is missing or just repeats the headline, the lead of the article body is used
  instead. News is written inverted-pyramid, so the headline and lead carry the
  who/what/where of the event. Deeper body text adds topic noise (background
  paragraphs, "also read" links) that makes DIFFERENT stories about the same
  subject look alike — the opposite of what clustering needs.

MODEL CHOICE — measured, not assumed:
  tools/cluster_eval/ calibrates candidate models on hand-labelled pairs of
  live Indian headlines (English + Hindi, cross-lingual included). Results and
  the decision live in docs/CLUSTERING.md. Any model in MODEL_PROFILES can be
  selected with CLUSTER_EMBEDDING_MODEL; the DB stores the model name on every
  cluster, so switching models never mixes incompatible vector spaces.

BATCHING:
  embed_texts() encodes in mini-batches — on a 2-vCPU runner this is ~4-6×
  faster than one call per article, which is why the runner embeds the whole
  claimed batch up front instead of inside the per-article loop.

INSTALL:
  pip install sentence-transformers   (torch CPU wheel installed separately)
"""

from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass

from enrichment.config import CLUSTER_EMBEDDING_MODEL, EMBED_BATCH_SIZE, EMBED_MAX_CHARS

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL PROFILES
#
# prefix:          text prepended before encoding. E5's model card recommends
#                  "query: " for symmetric tasks, but on our labelled Indian
#                  headline pairs the bare text separates same-story pairs
#                  better (ROC-AUC 0.930 vs 0.915, best F1 0.725 vs 0.698 —
#                  docs/CLUSTERING.md §Calibration), so no prefix is used.
# max_seq_length:  tokens kept by the tokenizer. 128 covers headline + lead
#                  with room to spare and keeps CPU cost low.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelProfile:
    model_id: str
    dimension: int
    prefix: str = ""
    max_seq_length: int = 128


MODEL_PROFILES: dict[str, ModelProfile] = {
    "sentence-transformers/LaBSE": ModelProfile(
        "sentence-transformers/LaBSE", dimension=768, prefix="", max_seq_length=128,
    ),
    "intfloat/multilingual-e5-base": ModelProfile(
        "intfloat/multilingual-e5-base", dimension=768, prefix="", max_seq_length=128,
    ),
}

# The SQL schema stores halfvec(768) / vector(768) — every profile must match.
EMBEDDING_DIMENSION = 768


def get_profile(model_id: str = CLUSTER_EMBEDDING_MODEL) -> ModelProfile:
    """Return the profile for model_id, or raise with the list of valid choices."""
    try:
        return MODEL_PROFILES[model_id]
    except KeyError:
        raise ValueError(
            f"Unknown CLUSTER_EMBEDDING_MODEL '{model_id}'. "
            f"Choose one of: {', '.join(sorted(MODEL_PROFILES))}"
        ) from None


# ─────────────────────────────────────────────────────────────────────────────
# TEXT CONSTRUCTION — pure, deterministic, unit-tested
# ─────────────────────────────────────────────────────────────────────────────

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
# Sentence end: Latin punctuation or the Devanagari danda (।).
_SENT_END_RE = re.compile(r"(?<=[.!?।])\s+")
_DESC_CHARS = 300
_MIN_DESC_CHARS = 40     # shorter than this, a description adds nothing


def _clean(text: str | None) -> str:
    """Strip tags, decode entities (&amp; → &), collapse whitespace."""
    if not text:
        return ""
    return _WS_RE.sub(" ", html.unescape(_TAG_RE.sub(" ", text))).strip()


def _normalised_words(text: str) -> str:
    return re.sub(r"[^\w]+", " ", text.lower()).strip()


def _repeats_title(title: str, desc: str) -> bool:
    """True when the description is just the headline again (very common)."""
    t, d = _normalised_words(title), _normalised_words(desc)
    return bool(t) and (d == t or d.startswith(t) or t.startswith(d))


def _lead(full_text: str, max_chars: int) -> str:
    """First sentences of the body up to max_chars, never cut mid-word."""
    body = _clean(full_text)
    if not body:
        return ""
    out = ""
    for sent in _SENT_END_RE.split(body):
        candidate = f"{out} {sent}".strip()
        if len(candidate) > max_chars:
            break
        out = candidate
    return out or _truncate_words(body, max_chars)


def _truncate_words(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    space = cut.rfind(" ")
    return cut[:space] if space > max_chars // 2 else cut


def build_embedding_text(
    title: str | None,
    description: str | None,
    full_text: str | None,
    max_chars: int = EMBED_MAX_CHARS,
) -> str:
    """
    Build the text that represents an article for clustering.

      "<headline>. <description or body lead>"   (≤ max_chars, word boundary)

    Returns "" when there is nothing to embed (the caller then skips
    clustering for that article rather than embedding noise).
    """
    t = _clean(title)
    d = _clean(description)
    context = ""
    if len(d) >= _MIN_DESC_CHARS and not _repeats_title(t, d):
        context = _truncate_words(d, _DESC_CHARS)
    else:
        context = _lead(full_text or "", _DESC_CHARS)
        if context and _repeats_title(t, context):
            context = ""

    if t and context:
        sep = "" if t.endswith((".", "!", "?", "।", ":")) else "."
        text = f"{t}{sep} {context}"
    else:
        text = t or context
    return _truncate_words(text, max_chars)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL — loaded once per process
# ─────────────────────────────────────────────────────────────────────────────

_model = None
_model_id: str | None = None


def _get_model(model_id: str):
    global _model, _model_id
    if _model is None or _model_id != model_id:
        profile = get_profile(model_id)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            ) from None
        log.info("Loading embedding model '%s'...", model_id)
        model = SentenceTransformer(model_id, device="cpu")
        model.max_seq_length = profile.max_seq_length
        dim = model.get_sentence_embedding_dimension()
        if dim != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Embedding model '{model_id}' produces {dim}-dim vectors; "
                f"the database schema stores {EMBEDDING_DIMENSION}-dim vectors."
            )
        _model, _model_id = model, model_id
        log.info("Embedding model loaded (dim=%d, max_seq_length=%d)", dim, profile.max_seq_length)
    return _model


def embed_texts(
    texts: list[str],
    model_id: str = CLUSTER_EMBEDDING_MODEL,
    batch_size: int = EMBED_BATCH_SIZE,
) -> list[list[float] | None]:
    """
    Encode texts into L2-normalised vectors, preserving input order.

    Empty texts map to None (nothing to cluster on). If the model fails, every
    entry is None and the caller skips clustering for this batch — enrichment
    of the articles themselves must never fail because of clustering.
    """
    out: list[list[float] | None] = [None] * len(texts)
    idx = [i for i, t in enumerate(texts) if t and t.strip()]
    if not idx:
        return out
    profile = get_profile(model_id)
    try:
        model = _get_model(model_id)
        vectors = model.encode(
            [profile.prefix + texts[i] for i in idx],
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    except Exception as e:
        log.error("Embedding batch of %d failed: %s — clustering skipped for it", len(idx), e)
        return out
    for i, vec in zip(idx, vectors):
        values = [float(v) for v in vec]
        # A zero / NaN vector would poison cosine maths in SQL.
        if any(v != v for v in values) or not any(values):
            continue
        out[i] = values
    return out


def to_pgvector(vector: list[float]) -> str:
    """Format a float list as a pgvector literal: '[0.012345,-0.2,…]'."""
    return "[" + ",".join(f"{v:.6f}" for v in vector) + "]"
