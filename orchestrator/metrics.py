from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

import os
import sys

import numpy as np

# Ensure `Statistics_and_Seq2Seq/` is importable when running as a module or script.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import all_functions


MZ = Literal["skip", "epsilon"]


@dataclass
class MetricConfig:
    mape_zero: MZ = "skip"
    mape_epsilon: float = 1e-8


def mape_safe(y_true: np.ndarray, y_pred: np.ndarray, cfg: MetricConfig) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    denom = np.abs(y_true)
    if cfg.mape_zero == "skip":
        mask = denom > cfg.mape_epsilon
        if not np.any(mask):
            return float("nan")
        return float(np.nanmean(np.abs(y_pred[mask] - y_true[mask]) / denom[mask]))

    denom = np.maximum(denom, cfg.mape_epsilon)
    return float(np.nanmean(np.abs(y_pred - y_true) / denom))


def smape_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_pred) + np.abs(y_true)
    denom = np.where(denom == 0, np.nan, denom)
    return float(np.nanmean(2.0 * np.abs(y_pred - y_true) / denom))


def rmse_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.nanmean((y_pred - y_true) ** 2)))


def pocid_within_sequence(y_true_seq: np.ndarray, y_pred_seq: np.ndarray) -> float:
    """POCID computed within a single horizon-path sequence (window-level).

    Matches the current project convention: `all_functions.pocid` compares
    consecutive steps inside the provided vectors.
    """

    y_true_seq = np.asarray(y_true_seq, dtype=float)
    y_pred_seq = np.asarray(y_pred_seq, dtype=float)
    if len(y_true_seq) < 2 or len(y_pred_seq) < 2:
        return float("nan")
    min_len = min(len(y_true_seq), len(y_pred_seq))
    return float(all_functions.pocid(y_true_seq[:min_len], y_pred_seq[:min_len]))


def compute_metrics_1d(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cfg: MetricConfig,
) -> Dict[str, float]:
    return {
        "MAPE": mape_safe(y_true, y_pred, cfg),
        "SMAPE": smape_safe(y_true, y_pred),
        "RMSE": rmse_safe(y_true, y_pred),
    }
