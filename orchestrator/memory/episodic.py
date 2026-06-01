"""SQLite-backed episodic memory for V5 RAG retrieval.

Each processed series leaves a row with its features (catch22 + classics), the method the
Selector chose, the resulting score on the final test (when available), and the scores of
every menu method on validation. The Selector queries the k nearest past episodes by
feature-space distance and uses them as in-context examples (CER, arXiv:2506.06698; Self-Gen
ICL, arXiv:2505.00234).

Schema deliberately stores raw feature vectors as JSON so the schema doesn't break when new
catch22 features are added. Retrieval reconstructs the vector at query time.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np


_DEFAULT_DB_PATH = os.environ.get("V5_MEMORY_DB", "./memory/v5_episodic.db")


# Feature keys used for nearest-neighbor distance. Anything in compute_series_features that
# starts with c22_ + this fixed list of classics. Fixed schema → reproducible distances.
_FEATURE_KEYS_CLASSIC = [
    "trend_strength",
    "seasonal_strength",
    "spectral_entropy",
    "forecastability",
    "hurst",
    "adf_pvalue",
    "variance_ratio_halves",
    "cv",
]


def _features_to_vector(features: Dict[str, Any]) -> np.ndarray:
    """Build a fixed-order numeric vector from a features dict. catch22 keys come first
    (alphabetical for stability), then the classic keys in their fixed order. Missing values
    become NaN so distance computation can ignore them via nanmean.
    """
    if not isinstance(features, dict):
        return np.array([], dtype=float)
    c22_keys = sorted(k for k in features.keys() if k.startswith("c22_"))
    parts: List[float] = []
    for k in c22_keys + _FEATURE_KEYS_CLASSIC:
        v = features.get(k)
        try:
            parts.append(float(v) if v is not None else float("nan"))
        except Exception:
            parts.append(float("nan"))
    return np.array(parts, dtype=float)


def _zscore_safe(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values, dtype=float)
    finite_std = np.where(np.isfinite(std) & (std > 1e-9), std, 1.0)
    finite_mean = np.where(np.isfinite(mean), mean, 0.0)
    diff = values - finite_mean
    diff = np.where(np.isfinite(diff), diff, 0.0)
    out = diff / finite_std
    return out


class EpisodicMemory:
    """Singleton-style episodic store. Use `EpisodicMemory.get_default()` everywhere to share
    one connection across the pipeline. Persistence: ./memory/v5_episodic.db (overridable via
    V5_MEMORY_DB env var or explicit `db_path`).
    """

    _instance: Optional["EpisodicMemory"] = None
    _lock = threading.Lock()

    def __init__(self, db_path: str = _DEFAULT_DB_PATH):
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_schema()

    # ── Class-level singleton accessor ───────────────────────────────────────
    @classmethod
    def get_default(cls, db_path: Optional[str] = None) -> Optional["EpisodicMemory"]:
        """Return the shared instance, creating it on first call. Returns None on hard error
        so callers can degrade gracefully (memory is optional)."""
        try:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(db_path or _DEFAULT_DB_PATH)
                elif db_path and db_path != cls._instance.db_path:
                    cls._instance = cls(db_path)
            return cls._instance
        except Exception:
            return None

    # ── Schema ───────────────────────────────────────────────────────────────
    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset TEXT NOT NULL,
                series_idx INTEGER NOT NULL,
                features_json TEXT NOT NULL,
                chosen_method TEXT NOT NULL,
                chosen_score REAL,
                method_scores_json TEXT,
                series_type TEXT,
                disagreement REAL,
                confidence TEXT,
                narrative TEXT,
                timestamp TEXT,
                UNIQUE(dataset, series_idx)
            );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_dataset ON episodes(dataset);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_method ON episodes(chosen_method);")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS procedural_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                condition_json TEXT NOT NULL,
                default_method TEXT NOT NULL,
                support_n INTEGER NOT NULL,
                win_rate REAL NOT NULL,
                avg_delta REAL,
                active INTEGER NOT NULL DEFAULT 1,
                updated_at TEXT
            );
        """)
        self._conn.commit()

    # ── Write ────────────────────────────────────────────────────────────────
    def add_episode(
        self,
        dataset: str,
        series_idx: int,
        features: Dict[str, Any],
        chosen_method: str,
        chosen_score: float = None,
        method_scores: Dict[str, Dict[str, float]] = None,
        series_type: str = "",
        disagreement: float = None,
        confidence: str = "",
        narrative: str = "",
    ) -> int:
        """Append (or replace by (dataset, series_idx)) one episode. Idempotent within a run."""
        cur = self._conn.cursor()
        cur.execute("""
            INSERT OR REPLACE INTO episodes
                (dataset, series_idx, features_json, chosen_method, chosen_score,
                 method_scores_json, series_type, disagreement, confidence, narrative, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(dataset),
            int(series_idx),
            json.dumps(features, default=str),
            str(chosen_method),
            float(chosen_score) if chosen_score is not None and np.isfinite(chosen_score) else None,
            json.dumps(method_scores or {}, default=str),
            str(series_type or ""),
            float(disagreement) if disagreement is not None and np.isfinite(disagreement) else None,
            str(confidence or ""),
            str(narrative or ""),
            datetime.utcnow().isoformat(timespec="seconds"),
        ))
        self._conn.commit()
        return int(cur.lastrowid)

    # ── Read / Query ─────────────────────────────────────────────────────────
    def _load_all(self, dataset: Optional[str] = None, exclude_series_idx: Optional[int] = None) -> List[Dict[str, Any]]:
        cur = self._conn.cursor()
        if dataset:
            cur.execute("SELECT id, dataset, series_idx, features_json, chosen_method, chosen_score, method_scores_json, series_type FROM episodes WHERE dataset = ?", (str(dataset),))
        else:
            cur.execute("SELECT id, dataset, series_idx, features_json, chosen_method, chosen_score, method_scores_json, series_type FROM episodes")
        rows = []
        for rid, ds, six, fj, m, sc, ms, st in cur.fetchall():
            if exclude_series_idx is not None and ds == dataset and int(six) == int(exclude_series_idx):
                continue
            try:
                feats = json.loads(fj) if fj else {}
            except Exception:
                feats = {}
            try:
                m_scores = json.loads(ms) if ms else {}
            except Exception:
                m_scores = {}
            rows.append({
                "id": int(rid),
                "dataset": ds,
                "series_idx": int(six),
                "features": feats,
                "chosen_method": m,
                "chosen_score": sc,
                "method_scores": m_scores,
                "series_type": st,
            })
        return rows

    def query_nearest(
        self,
        features: Dict[str, Any],
        k: int = 5,
        dataset: Optional[str] = None,
        exclude_series_idx: Optional[int] = None,
        cross_dataset_fallback: bool = True,
    ) -> List[Dict[str, Any]]:
        """Return up to `k` nearest past episodes by z-scored Euclidean distance over the
        feature vector. Restricts to `dataset` first; if fewer than k found, optionally
        widens to all datasets (`cross_dataset_fallback=True`).
        """
        in_vec = _features_to_vector(features)
        if in_vec.size == 0:
            return []

        # First pass: same dataset, exclude current series
        rows = self._load_all(dataset=dataset, exclude_series_idx=exclude_series_idx)
        if len(rows) < k and cross_dataset_fallback:
            rows = self._load_all(dataset=None)
            # exclude current (dataset, series_idx) explicitly
            if exclude_series_idx is not None and dataset is not None:
                rows = [r for r in rows if not (r["dataset"] == dataset and r["series_idx"] == int(exclude_series_idx))]
        if not rows:
            return []

        # Build feature matrix from rows (align on the SAME keys as in_vec by re-extracting)
        all_vecs = np.array([_features_to_vector(r["features"]) for r in rows], dtype=float)
        if all_vecs.size == 0:
            return []
        # Pad/truncate to align length (catch22 set is fixed, so this should be a no-op)
        n_dim = max(in_vec.size, all_vecs.shape[1])
        if in_vec.size < n_dim:
            in_vec = np.concatenate([in_vec, np.full(n_dim - in_vec.size, np.nan)])
        if all_vecs.shape[1] < n_dim:
            pad = np.full((all_vecs.shape[0], n_dim - all_vecs.shape[1]), np.nan)
            all_vecs = np.concatenate([all_vecs, pad], axis=1)

        # Z-score normalize using the corpus statistics (ignores NaN)
        mean = np.nanmean(all_vecs, axis=0)
        std = np.nanstd(all_vecs, axis=0)
        std = np.where(np.isfinite(std) & (std > 1e-9), std, 1.0)
        in_z = _zscore_safe(in_vec, mean, std)
        all_z = _zscore_safe(all_vecs, mean, std)

        # Euclidean distance over the feature space (NaN treated as 0 deviation)
        diff = all_z - in_z[None, :]
        diff = np.where(np.isfinite(diff), diff, 0.0)
        dists = np.sqrt(np.sum(diff ** 2, axis=1))

        order = np.argsort(dists)
        out: List[Dict[str, Any]] = []
        for idx in order[: max(0, int(k))]:
            r = rows[int(idx)]
            chosen_score = r.get("chosen_score")
            ms = r.get("method_scores") or {}
            # Build delta vs median (if available in the stored scores)
            delta_vs_median = None
            try:
                cm = ms.get(r["chosen_method"], {}).get("composite")
                md = ms.get("simple_median", {}).get("composite")
                if cm is not None and md is not None and np.isfinite(cm) and np.isfinite(md):
                    delta_vs_median = float(cm - md)
            except Exception:
                pass
            out.append({
                "neighbor_id": r["id"],
                "dataset": r["dataset"],
                "series_idx": r["series_idx"],
                "distance": float(dists[int(idx)]),
                "chosen_method": r["chosen_method"],
                "chosen_score": float(chosen_score) if chosen_score is not None else None,
                "series_type": r.get("series_type", ""),
                "delta_vs_median_on_validation": delta_vs_median,
            })
        return out

    def applicable_rules(self, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return active procedural rules whose `condition` matches the given features.
        Rules are produced by reflection (later sprint). For Sprint-C we keep an empty list
        unless something has been written manually; the Selector already has a sensible
        default when no rules apply.
        """
        cur = self._conn.cursor()
        cur.execute("SELECT id, condition_json, default_method, support_n, win_rate, avg_delta FROM procedural_rules WHERE active = 1")
        out = []
        for rid, cj, m, n, wr, dlt in cur.fetchall():
            try:
                cond = json.loads(cj) if cj else {}
            except Exception:
                cond = {}
            if _condition_matches(cond, features):
                out.append({
                    "rule_id": int(rid),
                    "condition": cond,
                    "default_method": m,
                    "support_n": int(n),
                    "win_rate": float(wr),
                    "avg_delta_vs_median": float(dlt) if dlt is not None else None,
                })
        return out

    # ── Diagnostics ──────────────────────────────────────────────────────────
    def count(self, dataset: Optional[str] = None) -> int:
        cur = self._conn.cursor()
        if dataset:
            cur.execute("SELECT COUNT(*) FROM episodes WHERE dataset = ?", (str(dataset),))
        else:
            cur.execute("SELECT COUNT(*) FROM episodes")
        return int(cur.fetchone()[0])

    def method_distribution(self, dataset: Optional[str] = None) -> Dict[str, int]:
        cur = self._conn.cursor()
        if dataset:
            cur.execute("SELECT chosen_method, COUNT(*) FROM episodes WHERE dataset = ? GROUP BY chosen_method", (str(dataset),))
        else:
            cur.execute("SELECT chosen_method, COUNT(*) FROM episodes GROUP BY chosen_method")
        return {m: int(c) for m, c in cur.fetchall()}


def _condition_matches(condition: Dict[str, Any], features: Dict[str, Any]) -> bool:
    """Evaluate a simple condition like {"seasonal_strength": {"gt": 0.7}, "series_type": "positive_only"}
    against features dict. Returns True if ALL keys satisfy their predicate.
    """
    if not isinstance(condition, dict) or not condition:
        return False
    for k, predicate in condition.items():
        v = features.get(k)
        if isinstance(predicate, dict):
            for op, target in predicate.items():
                try:
                    fv = float(v) if v is not None else float("nan")
                    if op == "gt" and not (fv > float(target)):
                        return False
                    elif op == "lt" and not (fv < float(target)):
                        return False
                    elif op == "ge" and not (fv >= float(target)):
                        return False
                    elif op == "le" and not (fv <= float(target)):
                        return False
                    elif op == "eq" and not (fv == float(target)):
                        return False
                except Exception:
                    return False
        else:
            if v != predicate:
                return False
    return True
