from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from agent.context import get_context


@dataclass
class ValidationData:
    """Normalized multi-step validation data."""

    y_true: np.ndarray  # shape (n_windows, horizon)
    y_preds: np.ndarray  # shape (n_windows, n_models, horizon)
    model_names: List[str]

    @property
    def n_windows(self) -> int:
        return int(self.y_true.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.y_true.shape[1])

    @property
    def n_models(self) -> int:
        return int(self.y_preds.shape[1])


def _to_2d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


def load_validation_from_context() -> ValidationData:
    """Reads `context['all_validations']` and returns normalized arrays.

    Expected shape (as used in `agent/context.py`):
        all_validations = {
            'predictions': [ {model: [h1, h2, ...], ...}, ... ],
            'test':        [ [h1, h2, ...], ... ]
        }

    Notes:
        - Supports variable horizon lengths by truncating each window to the minimum
          length across (test, all model preds) for that window.
        - Requires all windows to share at least one common model set; uses the
          model list from window 0.
    """

    all_validations = get_context("all_validations")
    if not all_validations:
        raise RuntimeError(
            "all_validations not found in CONTEXT_MEMORY. Call init_context() and generate_all_validations_context() first."
        )

    predictions_list = all_validations.get("predictions", [])
    test_list = all_validations.get("test", [])

    if not predictions_list or not test_list:
        raise RuntimeError("all_validations has empty predictions/test")

    n_windows = min(len(predictions_list), len(test_list))
    first_window = predictions_list[0]
    if not isinstance(first_window, dict) or not first_window:
        raise RuntimeError("all_validations['predictions'][0] is empty")

    model_names = list(first_window.keys())

    # Determine a consistent horizon across windows
    horizons: List[int] = []
    for i in range(n_windows):
        y_t = test_list[i]
        w_preds = predictions_list[i]
        if y_t is None or w_preds is None:
            continue
        lengths = [len(y_t)]
        for m in model_names:
            if m in w_preds and w_preds[m] is not None:
                lengths.append(len(w_preds[m]))
        horizons.append(min(lengths) if lengths else 0)

    horizon = min([h for h in horizons if h > 0], default=0)
    if horizon <= 0:
        raise RuntimeError("Could not infer a valid horizon from validation windows")

    y_true = np.full((n_windows, horizon), np.nan, dtype=float)
    y_preds = np.full((n_windows, len(model_names), horizon), np.nan, dtype=float)

    for i in range(n_windows):
        y_t = np.asarray(test_list[i], dtype=float)[:horizon]
        y_true[i, :] = y_t
        w_preds = predictions_list[i]
        for j, m in enumerate(model_names):
            if m not in w_preds:
                continue
            arr = np.asarray(w_preds[m], dtype=float)[:horizon]
            y_preds[i, j, :] = arr

    return ValidationData(y_true=y_true, y_preds=y_preds, model_names=model_names)
