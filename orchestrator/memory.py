"""Cross-dataset experiential memory for the multi-agent forecast pipeline.

Implements the long-horizon memory described in Tier D of `PIPELINE_DETAILS.md`.
Each completed pipeline run appends a single JSONL record to
`orchestrator/memory/runs.jsonl` (path is overridable).  Subsequent runs
retrieve the *k* most similar past records (cosine similarity over
FFORMA-style meta-feature vectors) and inject them as few-shot exemplars
in the Proposer / Council prompt.

Why memory at all?  This is the time-series specialisation of the
verbal-RL / experiential-learning pattern in Reflexion (Shinn et al.,
2023, NeurIPS) and ExpeL (Zhao et al., 2024, AAAI), and a natural online
analogue of the offline meta-learning of FFORMA (Montero-Manso et al.,
2020, IJF 36(1):86-92).  No paper in the LLM-agent forecasting literature
has, to our knowledge, fused FFORMA-style features with online
in-context retrieval.

The store is intentionally:
- **Append-only** (no rewrites; trivially auditable).
- **Schema-stable** (a frozen list of feature names is recorded with each
  entry so vector alignment survives feature-set additions).
- **Deterministic** (no randomness in retrieval; results are
  reproducible given the same JSONL file).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from orchestrator.feature_extractor import feature_vector


EPS = 1e-12

# Default location: under the project root.  Callers may override.
DEFAULT_MEMORY_PATH = (
    Path(__file__).resolve().parent / "memory" / "runs.jsonl"
)


# ──────────────────────────────────────────────────────────────────────────
# Record schema
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class MemoryRecord:
    """A single completed-run experiential record.

    All numeric fields are JSON-serialisable; vectors are kept as
    `Dict[str, float]` so the on-disk schema is human-readable and
    re-orderable.
    """

    dataset_id: str
    timestamp: str
    feature_vector: Dict[str, float]
    feature_schema: List[str]
    n_models: int
    n_windows: int
    horizon: int
    winning_strategy: str
    winning_strategy_type: str
    winning_params: Dict[str, Any]
    score: float
    score_vs_baseline_mean_pct: float
    debate_triggered: bool = False
    notes: str = ""

    def to_json_line(self) -> str:
        d = asdict(self)
        return json.dumps(d, separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MemoryRecord":
        return cls(
            dataset_id=str(d.get("dataset_id", "")),
            timestamp=str(d.get("timestamp", "")),
            feature_vector=dict(d.get("feature_vector", {})),
            feature_schema=list(d.get("feature_schema", [])),
            n_models=int(d.get("n_models", 0)),
            n_windows=int(d.get("n_windows", 0)),
            horizon=int(d.get("horizon", 0)),
            winning_strategy=str(d.get("winning_strategy", "")),
            winning_strategy_type=str(d.get("winning_strategy_type", "")),
            winning_params=dict(d.get("winning_params", {})),
            score=float(d.get("score", float("nan"))),
            score_vs_baseline_mean_pct=float(d.get("score_vs_baseline_mean_pct", float("nan"))),
            debate_triggered=bool(d.get("debate_triggered", False)),
            notes=str(d.get("notes", "")),
        )


# ──────────────────────────────────────────────────────────────────────────
# Store: append + read
# ──────────────────────────────────────────────────────────────────────────


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def append_run(
    record: MemoryRecord,
    path: Optional[Path] = None,
) -> Path:
    """Append a single record to the JSONL store."""

    p = Path(path) if path is not None else DEFAULT_MEMORY_PATH
    _ensure_parent(p)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(record.to_json_line())
        fh.write("\n")
    return p


def load_all(path: Optional[Path] = None) -> List[MemoryRecord]:
    """Load all records.  Bad lines are skipped with no error."""

    p = Path(path) if path is not None else DEFAULT_MEMORY_PATH
    if not p.exists():
        return []
    out: List[MemoryRecord] = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(MemoryRecord.from_dict(json.loads(line)))
            except Exception:
                continue
    return out


# ──────────────────────────────────────────────────────────────────────────
# Retrieval (cosine similarity, schema-aligned)
# ──────────────────────────────────────────────────────────────────────────


def _align(
    query: Dict[str, float],
    record: MemoryRecord,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return aligned numeric vectors for (query, record) on the union of keys.

    Missing values are treated as 0 *after* both vectors are mean-centred
    on the shared dimensions (so a missing field does not pull similarity
    artificially towards 0).
    """

    keys = sorted(set(query.keys()) | set(record.feature_vector.keys()))
    q = feature_vector(query, schema=keys)
    r = feature_vector(record.feature_vector, schema=keys)
    # NaN -> 0 after pairwise NaN-mask (so present-only dims drive the score)
    mask = (~np.isnan(q)) & (~np.isnan(r))
    if not np.any(mask):
        return np.zeros(0), np.zeros(0)
    return q[mask], r[mask]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < EPS or nb < EPS:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _z_normalise(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / np.where(std < EPS, 1.0, std)


@dataclass
class RetrievedExample:
    record: MemoryRecord
    similarity: float

    def to_few_shot(self) -> Dict[str, Any]:
        """Compact dict suitable for embedding in an LLM prompt."""

        return {
            "dataset_id": self.record.dataset_id,
            "similarity": round(self.similarity, 4),
            "n_windows": self.record.n_windows,
            "n_models": self.record.n_models,
            "horizon": self.record.horizon,
            "winning_strategy": self.record.winning_strategy,
            "winning_strategy_type": self.record.winning_strategy_type,
            "winning_params": self.record.winning_params,
            "score_vs_baseline_mean_pct": round(self.record.score_vs_baseline_mean_pct, 3),
            "debate_triggered": self.record.debate_triggered,
            "notes": self.record.notes,
        }


def retrieve_similar(
    query_features: Dict[str, float],
    k: int = 3,
    min_similarity: float = 0.6,
    exclude_dataset_ids: Optional[Iterable[str]] = None,
    path: Optional[Path] = None,
    z_normalise_corpus: bool = True,
) -> List[RetrievedExample]:
    """Return the *k* most similar past runs above `min_similarity`.

    Args:
        query_features: feature dict for the current dataset.
        k: number of results to return.
        min_similarity: cosine-similarity floor; results below are dropped.
        exclude_dataset_ids: skip records whose `dataset_id` is in this set
            (use this to prevent retrieving the very dataset being scored
            during a leave-one-out evaluation).
        path: override the JSONL store location.
        z_normalise_corpus: if True, vectors are z-scored using the mean
            and std of the corpus before cosine similarity.  This prevents
            high-magnitude features (e.g. `length`) from dominating.

    Returns:
        Sorted list of `RetrievedExample` (descending similarity).
    """

    records = load_all(path)
    if not records:
        return []
    excl = set(exclude_dataset_ids or [])
    records = [r for r in records if r.dataset_id not in excl]
    if not records:
        return []

    # Build a corpus matrix on the intersection of features
    corpus_keys = sorted(
        set.intersection(*[set(r.feature_vector.keys()) for r in records])
        & set(query_features.keys())
    )
    if not corpus_keys:
        return []

    Q = feature_vector(query_features, schema=corpus_keys)
    M = np.array([feature_vector(r.feature_vector, schema=corpus_keys) for r in records])

    # Drop columns where all corpus values are NaN
    col_mask = ~np.all(np.isnan(M), axis=0)
    if not np.any(col_mask):
        return []
    Q = Q[col_mask]
    M = M[:, col_mask]

    # Mean-impute remaining NaN with column mean
    col_mean = np.nanmean(M, axis=0)
    M = np.where(np.isnan(M), col_mean, M)
    Q = np.where(np.isnan(Q), col_mean, Q)

    if z_normalise_corpus and M.shape[0] >= 2:
        col_std = np.nanstd(M, axis=0)
        Q = _z_normalise(Q, col_mean, col_std)
        M = _z_normalise(M, col_mean, col_std)

    sims: List[Tuple[float, MemoryRecord]] = []
    for r, m in zip(records, M):
        s = _cosine(Q, m)
        if not np.isfinite(s):
            continue
        sims.append((s, r))

    sims.sort(key=lambda t: t[0], reverse=True)
    out = [
        RetrievedExample(record=r, similarity=float(s))
        for s, r in sims
        if s >= min_similarity
    ]
    return out[: max(0, int(k))]


# ──────────────────────────────────────────────────────────────────────────
# Convenience: build a record from pipeline outputs
# ──────────────────────────────────────────────────────────────────────────


def build_record(
    dataset_id: str,
    features: Dict[str, float],
    n_models: int,
    n_windows: int,
    horizon: int,
    winning_strategy: str,
    winning_strategy_type: str,
    winning_params: Dict[str, Any],
    score: float,
    score_vs_baseline_mean_pct: float,
    debate_triggered: bool = False,
    notes: str = "",
) -> MemoryRecord:
    """Helper to instantiate a `MemoryRecord` with a UTC ISO-8601 timestamp."""

    return MemoryRecord(
        dataset_id=dataset_id,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        feature_vector=dict(features),
        feature_schema=sorted(features.keys()),
        n_models=int(n_models),
        n_windows=int(n_windows),
        horizon=int(horizon),
        winning_strategy=str(winning_strategy),
        winning_strategy_type=str(winning_strategy_type),
        winning_params=dict(winning_params),
        score=float(score),
        score_vs_baseline_mean_pct=float(score_vs_baseline_mean_pct),
        debate_triggered=bool(debate_triggered),
        notes=str(notes),
    )
