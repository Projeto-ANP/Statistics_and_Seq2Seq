"""Reading the original `.tsf` series (Phase 0, part 1).

The forecast-combination pipeline consumes per-model result CSVs, which carry only
the forecast windows. `series_profile()` needs the actual historical series, which
lives in the original `.tsf` files, outside the repository (default
`../forecasting_datasets`).

Mapping `dataset_index` -> series was validated point by point against the result
CSVs before this module was written (see EXPLORACAO.md, section D3):

    ANP_MONTHLY          mes_11_venda_mensal.tsf   216 -> 182 after the zero filter
    NN5_WEEKLY_DATASET   nn5_weekly_dataset.tsf    direct positional mapping
    M4_WEEKLY_DATASET    m4_weekly_dataset.tsf     direct positional mapping
    ETTH1/2, ETTM1/2     ETTh*.tsf / ETTm*.tsf     direct positional mapping

`dataset_index` is the **positional** index of the row in the (possibly filtered)
frame, exactly as `run_tsf_regressors.py` produced it with `df.iloc[i]`.

Nothing here is trusted blindly: `verify_alignment` compares the tail of the loaded
series with the `test` column of the result CSV and raises on mismatch, so a wrong
`.tsf` can never be silently paired with another series' forecasts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np


DEFAULT_SOURCE_DIR = "../forecasting_datasets"


class SeriesAlignmentError(RuntimeError):
    """The loaded series does not match the forecasts recorded for that index."""


# ──────────────────────────────────────────────────────────────────────────────
# .tsf parsing
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class TsfFile:
    """Parsed Monash `.tsf` file."""

    attributes: List[str]
    rows: List[Dict[str, Any]]  # metadata keys + "series_value" (np.ndarray)
    frequency: str = ""
    horizon: Optional[int] = None
    relation: str = ""
    path: str = ""

    def __len__(self) -> int:
        return len(self.rows)


def parse_tsf(path: str) -> TsfFile:
    """Minimal, dependency-free Monash `.tsf` reader.

    Deliberately not using `streamfuels.DatasetLoader`: this module must stay
    importable in a plain numpy environment so the ingestion can be unit tested.
    Row order is preserved, which is what makes the positional `dataset_index`
    mapping valid.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"tsf file not found: {path}")

    attributes: List[str] = []
    rows: List[Dict[str, Any]] = []
    frequency, horizon, relation = "", None, ""
    in_data = False

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\n").rstrip("\r")
            if not line or line.startswith("#"):
                continue
            if not in_data and line.startswith("@"):
                lower = line.lower()
                if lower.startswith("@attribute"):
                    parts = line.split()
                    if len(parts) >= 2:
                        attributes.append(parts[1])
                elif lower.startswith("@frequency"):
                    frequency = line.split(maxsplit=1)[1].strip() if " " in line else ""
                elif lower.startswith("@horizon"):
                    try:
                        horizon = int(line.split(maxsplit=1)[1].strip())
                    except (ValueError, IndexError):
                        horizon = None
                elif lower.startswith("@relation"):
                    relation = line.split(maxsplit=1)[1].strip() if " " in line else ""
                elif lower.startswith("@data"):
                    in_data = True
                continue
            if not in_data:
                continue

            parts = line.split(":")
            if len(parts) <= len(attributes):
                continue
            meta = dict(zip(attributes, parts[: len(attributes)]))
            values_str = ":".join(parts[len(attributes) :])
            values = np.array(
                [np.nan if v.strip() in {"?", ""} else float(v) for v in values_str.split(",")],
                dtype=float,
            )
            meta["series_value"] = values
            rows.append(meta)

    if not rows:
        raise ValueError(f"no data rows found in {path}")
    return TsfFile(
        attributes=attributes,
        rows=rows,
        frequency=frequency,
        horizon=horizon,
        relation=relation,
        path=path,
    )


# ──────────────────────────────────────────────────────────────────────────────
# series filters used when the base models were generated
# ──────────────────────────────────────────────────────────────────────────────


def drop_zero_windows(values: np.ndarray, window_size: int = 24) -> bool:
    """True when the series should be dropped: >50% zeros in any window.

    Reproduces the `should_remove` helper that is commented out at
    `run_tsf_regressors.py:820-828`. Validated: it takes the ANP file from 216 to
    exactly the 182 series present in the result CSVs.
    """
    v = np.asarray(values, dtype=float)
    for i in range(0, v.size, int(window_size)):
        chunk = v[i : i + int(window_size)]
        if chunk.size and float((chunk == 0).mean()) > 0.5:
            return True
    return False


#: Filters tried, in order, when the raw row count does not match the results.
KNOWN_FILTERS: List[tuple] = [
    ("drop_zero_windows_24", lambda v: drop_zero_windows(v, 24)),
    ("drop_zero_windows_12", lambda v: drop_zero_windows(v, 12)),
]


# ──────────────────────────────────────────────────────────────────────────────
# source
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class SeriesSource:
    """Series of one dataset, indexed exactly like `dataset_index`."""

    values: List[np.ndarray]
    metadata: List[Dict[str, Any]] = field(default_factory=list)
    frequency: str = ""
    declared_horizon: Optional[int] = None
    path: str = ""
    filter_applied: str = "none"

    @property
    def n_series(self) -> int:
        return len(self.values)

    def series(self, dataset_index: int) -> np.ndarray:
        i = int(dataset_index)
        if not (0 <= i < self.n_series):
            raise IndexError(
                f"dataset_index={i} outside [0, {self.n_series - 1}] for {os.path.basename(self.path)}"
            )
        return self.values[i]

    def info(self) -> Dict[str, Any]:
        return {
            "file": os.path.basename(self.path),
            "n_series": self.n_series,
            "frequency": self.frequency,
            "declared_horizon": self.declared_horizon,
            "filter_applied": self.filter_applied,
        }


def load_series_source(
    source_file: str,
    n_expected_series: Optional[int] = None,
    source_dir: str = DEFAULT_SOURCE_DIR,
) -> SeriesSource:
    """Loads a `.tsf` and aligns its row count with what the results expect.

    When `n_expected_series` is given and the raw count differs, the known
    generation filters are tried in order. If none reproduces the expected count,
    this raises instead of guessing — a wrong alignment would pair each
    `dataset_index` with someone else's series.
    """
    path = source_file if os.path.isabs(source_file) else os.path.join(source_dir, source_file)
    tsf = parse_tsf(path)

    rows = tsf.rows
    filter_applied = "none"

    if n_expected_series is not None and len(rows) != int(n_expected_series):
        matched = False
        for name, predicate in KNOWN_FILTERS:
            kept = [r for r in tsf.rows if not predicate(r["series_value"])]
            if len(kept) == int(n_expected_series):
                rows, filter_applied, matched = kept, name, True
                break
        if not matched:
            raise SeriesAlignmentError(
                f"{os.path.basename(path)} has {len(tsf.rows)} series but the results expect "
                f"{n_expected_series}, and no known filter reconciles them "
                f"(tried: {[n for n, _ in KNOWN_FILTERS]}).\n"
                "This usually means the .tsf on disk is not the one used to generate the "
                "forecasts — e.g. the reduced single-series ETT files. See EXPLORACAO.md, D3."
            )

    return SeriesSource(
        values=[r["series_value"] for r in rows],
        metadata=[{k: v for k, v in r.items() if k != "series_value"} for r in rows],
        frequency=tsf.frequency,
        declared_horizon=tsf.horizon,
        path=path,
        filter_applied=filter_applied,
    )


#: Aggregations tried when the raw `.tsf` tail does not match the results. The ETT
#: forecasts were generated on a MEAN-RESAMPLED series, not the raw file: ETTM2.tsf
#: is 15-minute data, and the recorded `test` column matches the raw series
#: averaged in blocks of 2 (i.e. 30-minute bars) to 1e-6. Reading the raw file and
#: refusing was correct-but-unhelpful — the file IS the right one, at a different
#: resolution. Only mean-aggregation by a small integer factor is attempted, and
#: only after the raw comparison fails, so nothing that already worked changes.
RESAMPLE_FACTORS: Tuple[int, ...] = (2, 3, 4, 6, 12)


def mean_resample(series: np.ndarray, factor: int) -> np.ndarray:
    """Non-overlapping block mean. Trailing points that do not fill a block are
    dropped from the FRONT, so the tail — the part being verified — stays aligned
    with the end of the series."""
    series = np.asarray(series, dtype=float)
    factor = int(factor)
    if factor <= 1 or series.size < factor:
        return series
    usable = (series.size // factor) * factor
    return series[series.size - usable:].reshape(-1, factor).mean(axis=1)


def verify_alignment(
    series: np.ndarray,
    expected_tail: Sequence[float],
    dataset_index: int,
    source_name: str = "",
    rtol: float = 1e-4,
    atol: float = 1e-4,
    allow_resample: bool = True,
) -> Dict[str, Any]:
    """Guardrail: the tail of the series must equal the recorded test window.

    This is the check that makes it impossible to combine one series' forecasts
    with another series' profile. Raises `SeriesAlignmentError` on mismatch.

    `allow_resample` additionally accepts a series that matches after mean
    aggregation by one of `RESAMPLE_FACTORS` — the ETT case, where the forecasts
    were produced on 30-minute bars built from a 15-minute file. The factor used is
    returned in `resample_factor` so the series the agent profiles is the same one
    the models actually saw, and so the CSV records which resolution was used.
    """
    series = np.asarray(series, dtype=float)
    expected = np.asarray(expected_tail, dtype=float)
    h = expected.size

    if h == 0:
        raise SeriesAlignmentError(f"empty test window for dataset_index={dataset_index}")
    if series.size < h:
        raise SeriesAlignmentError(
            f"series for dataset_index={dataset_index} has {series.size} points, "
            f"fewer than the {h}-step test window"
        )

    tail = series[-h:]
    if allow_resample and not np.allclose(tail, expected, rtol=rtol, atol=atol, equal_nan=True):
        for factor in RESAMPLE_FACTORS:
            candidate = mean_resample(series, factor)
            if candidate.size < h:
                continue
            if np.allclose(candidate[-h:], expected, rtol=rtol, atol=atol, equal_nan=True):
                return {
                    "verified": True,
                    "horizon": int(h),
                    "n_points": int(candidate.size),
                    "max_abs_diff": float(np.nanmax(np.abs(candidate[-h:] - expected))),
                    "resample_factor": factor,
                    "series": candidate,
                }

    if not np.allclose(tail, expected, rtol=rtol, atol=atol, equal_nan=True):
        diff = np.abs(tail - expected)
        worst = int(np.nanargmax(diff))
        raise SeriesAlignmentError(
            f"series mismatch at dataset_index={dataset_index}"
            + (f" (source {source_name})" if source_name else "")
            + f": the last {h} points of the .tsf do not match the `test` column of the "
            f"result CSV. Largest gap at step {worst}: "
            f"tsf={tail[worst]:.6g} vs results={expected[worst]:.6g}. "
            "The .tsf is probably not the file used to generate the forecasts."
        )

    return {
        "verified": True,
        "horizon": int(h),
        "n_points": int(series.size),
        "max_abs_diff": float(np.nanmax(np.abs(tail - expected))),
        "resample_factor": 1,
        "series": series,
    }
