"""Phase 0 — ingestion (pure code, no LLM).

Reads the per-model result CSVs, assembles the validation windows and the test
forecasts, loads the historical series from the original `.tsf`, and returns a
ready `ReactState`.

The input format is **preserved exactly** as the existing pipeline produces it:

    ./timeseries/mestrado/resultados/<MODEL>/normal/<DATASET>.csv     sep=";"
    columns: dataset_index; horizon; regressor; mape; pocid; smape; rmse;
             msmape; mae; test; predictions; start_test; final_test

Rows are sorted by `start_test`; the **last** row is the blind test window and the
preceding ones are validation windows. Note the difference from the legacy
`generate_all_validations_context`, which took `train_window` and sliced
`iloc[-train_window:-1]`: here `n_windows` is the number of validation windows
directly, so `n_windows=3` is equivalent to the legacy `train_window=4`.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from orchestrator_react.config import ReactConfig
from orchestrator_react.data_source import (
    DEFAULT_SOURCE_DIR,
    SeriesSource,
    load_series_source,
    verify_alignment,
)
from orchestrator_react.state import ReactState


DEFAULT_RESULTS_DIR = "./timeseries/mestrado/resultados"

#: Combination baselines already computed on disk, in flat layout
#: `<results_dir>/<NAME>/<DATASET>.csv` (no `normal/` subfolder).
EXTERNAL_BASELINES = ("mean", "median", "dba", "ADE", "FFORMA")

_NUMBER = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


class IngestionError(RuntimeError):
    """The result CSVs are missing, inconsistent or misaligned."""


def extract_values(cell: Any) -> List[float]:
    """Parses the stringified arrays stored in `test` / `predictions`.

    Same regex the project has always used — the cells are numpy `repr` output and
    may contain line breaks and scientific notation.
    """
    if isinstance(cell, (list, tuple, np.ndarray)):
        return [float(v) for v in cell]
    if not isinstance(cell, str):
        return []
    return [float(v) for v in _NUMBER.findall(cell)]


def model_csv_path(model: str, dataset: str, results_dir: str = DEFAULT_RESULTS_DIR) -> str:
    return os.path.join(results_dir, model, "normal", f"{dataset}.csv")


def read_model_predictions(
    model: str, dataset: str, results_dir: str = DEFAULT_RESULTS_DIR
) -> pd.DataFrame:
    """Reads one model's result file, parsed and sorted by `start_test`."""
    path = model_csv_path(model, dataset, results_dir)
    if not os.path.exists(path):
        raise IngestionError(f"result file not found for model {model!r}: {path}")
    df = pd.read_csv(path, sep=";")
    for col in ("start_test", "final_test"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df.sort_values("start_test").reset_index(drop=True)


def count_series(dataset: str, model: str, results_dir: str = DEFAULT_RESULTS_DIR) -> int:
    """Number of distinct `dataset_index` values in a dataset's results."""
    path = model_csv_path(model, dataset, results_dir)
    if not os.path.exists(path):
        raise IngestionError(f"result file not found: {path}")
    return int(pd.read_csv(path, sep=";", usecols=["dataset_index"])["dataset_index"].nunique())


@dataclass
class IngestedSeries:
    """Everything Phase 0 produces for one series."""

    state: ReactState
    dataset: str
    dataset_index: int
    horizon: int
    #: Actual values of the blind test window, read from the reference model's row.
    #: The metrics in the CSV are computed from these, so they are carried here
    #: rather than re-read downstream.
    test_values: List[float] = field(default_factory=list)
    start_test: Any = None
    final_test: Any = None
    source_info: Dict[str, Any] = field(default_factory=dict)
    alignment: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "dataset_index": self.dataset_index,
            "horizon": self.horizon,
            "n_models": self.state.n_models,
            "n_validation_windows": self.state.n_windows,
            "test_values": len(self.test_values),
            "train_series_points": (
                0 if self.state.train_series is None else int(self.state.train_series.size)
            ),
            "source": self.source_info,
            "alignment": self.alignment,
            "warnings": self.warnings,
        }


def find_misaligned_models(
    models: Sequence[str],
    dataset: str,
    dataset_index: int,
    results_dir: str = DEFAULT_RESULTS_DIR,
    frames: Optional[Dict[str, pd.DataFrame]] = None,
    reference: Optional[str] = None,
    n_windows: int = 3,
) -> Dict[str, str]:
    """Models whose ACTUALS disagree with the reference model, and why.

    A forecast can only be combined with another if both are predicting the same
    points in time. On ETTM1 and ETTM2 five models — ONLY_CWT_catboost,
    ONLY_DWT_catboost, ONLY_DWT_rf, ONLY_FT_catboost, ONLY_FT_rf — were generated
    over a different window than the other fourteen (2018-06-25 20:00 at 15-minute
    steps versus 2018-06-24 20:00 at 30-minute steps), so their "test" column holds
    different numbers entirely. Averaging across that is not a combination, it is
    two unrelated quantities added together.

    Returns `{}` when everything lines up, which is the case on ANP and NN5.
    """
    frames = frames or load_dataset_frames(models, dataset, results_dir)
    reference = reference or models[0]
    ref = frames[reference]
    ref_rows = ref[ref["dataset_index"] == dataset_index].sort_values("start_test")
    if ref_rows.empty:
        return {}
    horizon = len(extract_values(ref_rows.iloc[-1]["test"]))

    out: Dict[str, str] = {}
    for m in models:
        if m == reference:
            continue
        df = frames[m]
        rows = df[df["dataset_index"] == dataset_index].sort_values("start_test")
        if len(rows) != len(ref_rows):
            out[m] = f"{len(rows)} windows vs {len(ref_rows)} in {reference!r}"
            continue
        for pos in range(-(n_windows + 1), 0):
            a = np.asarray(extract_values(rows.iloc[pos]["test"]), dtype=float)[:horizon]
            r = np.asarray(extract_values(ref_rows.iloc[pos]["test"]), dtype=float)[:horizon]
            if a.size == horizon and r.size == horizon and not np.allclose(
                a, r, rtol=1e-4, atol=1e-4, equal_nan=True
            ):
                out[m] = (
                    f"different actuals from {reference!r} "
                    f"(starts {rows.iloc[pos]['start_test']} vs {ref_rows.iloc[pos]['start_test']})"
                )
                break
    return out


def load_series(
    models: Sequence[str],
    dataset: str,
    dataset_index: int,
    source: Optional[SeriesSource] = None,
    config: Optional[ReactConfig] = None,
    results_dir: str = DEFAULT_RESULTS_DIR,
    source_file: Optional[str] = None,
    source_dir: str = DEFAULT_SOURCE_DIR,
    frames: Optional[Dict[str, pd.DataFrame]] = None,
    drop_models: Sequence[str] = (),
) -> IngestedSeries:
    """Builds the `ReactState` for one series.

    `drop_models` removes models from the pool before anything is read. Used for
    datasets where some models were generated over a different window than the
    rest (see `find_misaligned_models`); dropping them yields a smaller but
    coherent pool, which is the only kind that can be combined honestly.

    Args:
        models: pool model names (folder names under `results_dir`).
        dataset: results dataset name, e.g. "ANP_MONTHLY".
        dataset_index: series identifier inside the dataset.
        source: pre-loaded `.tsf` source. When None and `source_file` is given, it
            is loaded here — pass one in when looping over many series.
        source_file: `.tsf` file name, e.g. "mes_11_venda_mensal.tsf".
        frames: optional cache of already-read result CSVs, keyed by model.

    Raises:
        IngestionError on missing files, misaligned windows or disagreeing actuals.
    """
    config = config or ReactConfig()
    n_windows = int(config.n_validation_windows)
    if n_windows < 1:
        raise IngestionError("n_validation_windows must be >= 1")

    warnings: List[str] = []
    if drop_models:
        dropped = [m for m in models if m in set(drop_models)]
        models = [m for m in models if m not in set(drop_models)]
        if dropped:
            warnings.append(
                f"dropped {len(dropped)} model(s) whose windows do not match the rest "
                f"of the pool: {dropped}"
            )
    if not models:
        raise IngestionError("empty model pool")

    # ── per-model rows for this series ───────────────────────────────────────
    per_model: Dict[str, pd.DataFrame] = {}
    for m in models:
        df = frames[m] if frames and m in frames else read_model_predictions(m, dataset, results_dir)
        rows = df[df["dataset_index"] == int(dataset_index)].sort_values("start_test")
        if rows.empty:
            raise IngestionError(f"model {m!r} has no rows for dataset_index={dataset_index}")
        if len(rows) < n_windows + 1:
            raise IngestionError(
                f"model {m!r} has {len(rows)} windows for dataset_index={dataset_index}, "
                f"but {n_windows} validation windows + 1 test window are required"
            )
        per_model[m] = rows

    reference = models[0]
    ref_rows = per_model[reference]
    final_row = ref_rows.iloc[-1]
    final_test, start_test = final_row.get("final_test"), final_row.get("start_test")

    # Timestamp disagreement is a WARNING, not an error. What defines a window is
    # the data in it, not the label: the ETTH1 results are a real case where six
    # `ONLY_*` models wrote their index with freq="15min" instead of hourly, so the
    # same 24 observations are labelled 2016-12-29 in one file and 2018-06-26 in
    # another. The values are identical. Rejecting on the label would throw away a
    # perfectly usable pool; the hard gate is the value comparison below.
    timestamp_mismatch = [
        f"{m} ends at {rows.iloc[-1].get('final_test')}"
        for m, rows in per_model.items()
        if pd.notna(final_test)
        and pd.notna(rows.iloc[-1].get("final_test"))
        and rows.iloc[-1].get("final_test") != final_test
    ]
    if timestamp_mismatch:
        warnings.append(
            f"{len(timestamp_mismatch)} model(s) disagree with {reference!r} on the "
            f"final_test timestamp while holding identical data (a frequency-labelling "
            f"bug in those result files): {timestamp_mismatch[:4]}. "
            f"The reported final_test comes from {reference!r}."
        )

    # ── horizon: shortest common length across windows and models ────────────
    window_rows = list(range(-(n_windows + 1), 0))  # oldest validation .. test
    lengths: List[int] = []
    for m, rows in per_model.items():
        for pos in window_rows:
            row = rows.iloc[pos]
            lengths.append(len(extract_values(row["predictions"])))
            if pos != -1:
                lengths.append(len(extract_values(row["test"])))
    lengths.append(len(extract_values(final_row["test"])))
    horizon = int(min(v for v in lengths if v > 0)) if any(v > 0 for v in lengths) else 0
    if horizon <= 0:
        raise IngestionError(f"could not infer a horizon for dataset_index={dataset_index}")
    if len(set(lengths)) > 1:
        warnings.append(f"ragged window lengths {sorted(set(lengths))}; truncated to {horizon}")

    # ── validation windows (oldest -> newest) and test forecasts ─────────────
    y_true = np.full((n_windows, horizon), np.nan, dtype=float)
    y_preds = np.full((n_windows, len(models), horizon), np.nan, dtype=float)
    test_preds = np.full((len(models), horizon), np.nan, dtype=float)

    for w, pos in enumerate(range(-(n_windows + 1), -1)):
        ref_actual = np.asarray(extract_values(ref_rows.iloc[pos]["test"]), dtype=float)[:horizon]
        y_true[w, :] = ref_actual
        for j, m in enumerate(models):
            row = per_model[m].iloc[pos]
            actual = np.asarray(extract_values(row["test"]), dtype=float)[:horizon]
            if actual.size == horizon and not np.allclose(
                actual, ref_actual, rtol=1e-4, atol=1e-4, equal_nan=True
            ):
                raise IngestionError(
                    f"model {m!r} reports different actuals than {reference!r} on validation "
                    f"window {w} of dataset_index={dataset_index}: the windows are misaligned"
                )
            y_preds[w, j, :] = np.asarray(extract_values(row["predictions"]), dtype=float)[:horizon]

    ref_test_actual = np.asarray(extract_values(final_row["test"]), dtype=float)[:horizon]
    for j, m in enumerate(models):
        row = per_model[m].iloc[-1]
        actual = np.asarray(extract_values(row["test"]), dtype=float)[:horizon]
        # The blind window must be the same one for everybody. This is the check the
        # timestamp comparison used to stand in for, done on the data itself.
        if actual.size == horizon and not np.allclose(
            actual, ref_test_actual, rtol=1e-4, atol=1e-4, equal_nan=True
        ):
            raise IngestionError(
                f"model {m!r} reports different actuals than {reference!r} on the test "
                f"window of dataset_index={dataset_index}: these are not the same window"
            )
        test_preds[j, :] = np.asarray(extract_values(row["predictions"]), dtype=float)[:horizon]

    # ── historical series from the .tsf, with the alignment guardrail ────────
    train_series: Optional[np.ndarray] = None
    alignment: Dict[str, Any] = {"verified": False}
    source_info: Dict[str, Any] = {}

    if source is None and source_file:
        source = load_series_source(
            source_file,
            n_expected_series=count_series(dataset, reference, results_dir),
            source_dir=source_dir,
        )

    if source is not None:
        source_info = source.info()
        full = np.asarray(source.series(dataset_index), dtype=float)
        alignment = verify_alignment(
            full,
            expected_tail=extract_values(final_row["test"])[:horizon],
            dataset_index=int(dataset_index),
            source_name=source_info.get("file", ""),
        )
        # `verify_alignment` may have matched a MEAN-RESAMPLED version of the file
        # (the ETT case: 30-minute bars built from a 15-minute .tsf). Profile the
        # resolution the models were actually generated on, not the raw file, or
        # every seasonality and autocorrelation feature would describe a different
        # series than the forecasts do.
        full = np.asarray(alignment.get("series", full), dtype=float)
        factor = int(alignment.get("resample_factor", 1))
        if factor > 1:
            warnings.append(
                f"the .tsf is finer than the forecasts: matched after mean-resampling "
                f"by {factor} ({source_info.get('frequency')} -> blocks of {factor})"
            )
        # Same split the generation pipeline used: train = series[:-horizon].
        train_series = full[: full.size - horizon]
    else:
        warnings.append(
            "no .tsf source given: series_profile() will fall back to the validation windows"
        )

    freq = str(source_info.get("frequency") or "")
    state = ReactState(
        y_true=y_true,
        y_preds=y_preds,
        test_preds=test_preds,
        model_names=list(models),
        train_series=train_series,
        config=config,
        dataset_index=int(dataset_index),
        freq=freq,
    )

    return IngestedSeries(
        state=state,
        dataset=dataset,
        dataset_index=int(dataset_index),
        horizon=horizon,
        test_values=[float(v) for v in extract_values(final_row["test"])[:horizon]],
        start_test=start_test,
        final_test=final_test,
        source_info=source_info,
        alignment=alignment,
        warnings=warnings,
    )


def load_dataset_frames(
    models: Sequence[str], dataset: str, results_dir: str = DEFAULT_RESULTS_DIR
) -> Dict[str, pd.DataFrame]:
    """Reads every model's CSV once, to be reused across all series of a dataset."""
    return {m: read_model_predictions(m, dataset, results_dir) for m in models}


def read_external_baselines(
    dataset: str,
    dataset_index: int,
    results_dir: str = DEFAULT_RESULTS_DIR,
    names: Sequence[str] = EXTERNAL_BASELINES,
) -> Dict[str, Any]:
    """Metrics of the already-computed combination baselines, for the same series.

    Feeds the `baseline_results_json` CSV field. These files use the flat layout
    `<results_dir>/<NAME>/<DATASET>.csv` and were produced with the same 3
    validation windows, so they are directly comparable row by row.

    Missing baselines are reported rather than raised: FFORMA and ADE do not exist
    for every dataset.
    """
    out: Dict[str, Any] = {}
    for name in names:
        path = os.path.join(results_dir, name, f"{dataset}.csv")
        if not os.path.exists(path):
            out[name] = {"available": False, "reason": "file not found"}
            continue
        try:
            df = pd.read_csv(path, sep=";")
            rows = df[df["dataset_index"] == int(dataset_index)]
            if rows.empty:
                out[name] = {"available": False, "reason": "series not in file"}
                continue
            row = rows.iloc[-1]
            out[name] = {
                "available": True,
                "mape": _num(row.get("mape")),
                "pocid": _num(row.get("pocid")),
                "smape": _num(row.get("smape")),
                "rmse": _num(row.get("rmse")),
                "msmape": _num(row.get("msmape")),
                "mae": _num(row.get("mae")),
            }
        except Exception as exc:  # pragma: no cover - depends on files on disk
            out[name] = {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    return out


def _num(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None
