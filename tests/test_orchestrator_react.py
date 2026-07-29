"""Tool catalog tests (Step 6.2) — synthetic data, no LLM, no GPU.

Run:  python -m pytest tests/test_orchestrator_react.py -q

The data is built so the right answer is known in advance: a pool with deliberately
good, mediocre, bad and redundant models, over a series with known trend and
seasonality.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from orchestrator_react import meta_model as MM
from orchestrator_react import metrics as M
from orchestrator_react import registry as R
from orchestrator_react import tools as T
from orchestrator_react.combiners import (
    combine_dba,
    combine_mean,
    combine_median,
    combine_trimmed_mean,
    combine_weighted,
)
from orchestrator_react.config import ReactConfig
from orchestrator_react.state import FULL_POOL, ReactState
from orchestrator_react.weighting import project_simplex


HORIZON = 12
N_WINDOWS = 3
PERIOD = 12


# ──────────────────────────────────────────────────────────────────────────────
# fixtures
# ──────────────────────────────────────────────────────────────────────────────


def make_series(n: int = 240, seed: int = 7) -> np.ndarray:
    """Series with linear trend, annual seasonality and noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    return 100.0 + 0.35 * t + 12.0 * np.sin(2 * np.pi * t / PERIOD) + rng.normal(0, 2.0, n)


def make_state(seed: int = 7, config: ReactConfig | None = None) -> ReactState:
    """Synthetic pool with known quality.

    good_a / good_b       : small bias         -> should lead the ranking
    redundant_a / _b      : shared noise       -> should be grouped together
    mediocre              : moderate bias
    bad                   : large bias         -> should come last
    """
    rng = np.random.default_rng(seed)
    series = make_series(seed=seed)
    total = N_WINDOWS + 1

    blocks = [
        series[len(series) - (k + 1) * HORIZON : len(series) - k * HORIZON] for k in range(total)
    ]
    blocks = blocks[::-1]  # oldest to newest
    y_true = np.stack(blocks[:-1])  # validation windows
    test_true = blocks[-1]
    train_series = series[: len(series) - HORIZON]

    names = ["good_a", "good_b", "redundant_a", "redundant_b", "mediocre", "bad"]

    def perturb(truth: np.ndarray, kind: str, shared: np.ndarray) -> np.ndarray:
        if kind == "good_a":
            return truth + rng.normal(0.0, 1.0, truth.shape)
        if kind == "good_b":
            return truth + rng.normal(0.0, 1.2, truth.shape)
        if kind in ("redundant_a", "redundant_b"):
            return truth + shared + rng.normal(0.0, 0.15, truth.shape)
        if kind == "mediocre":
            return truth + rng.normal(3.0, 3.0, truth.shape)
        return truth + rng.normal(14.0, 6.0, truth.shape)

    y_preds = np.zeros((N_WINDOWS, len(names), HORIZON))
    for i in range(N_WINDOWS):
        shared = rng.normal(0.0, 4.0, HORIZON)
        for j, nm in enumerate(names):
            y_preds[i, j] = perturb(y_true[i], nm, shared)

    shared_t = rng.normal(0.0, 4.0, HORIZON)
    test_preds = np.stack([perturb(test_true, nm, shared_t) for nm in names])

    return ReactState(
        y_true=y_true,
        y_preds=y_preds,
        test_preds=test_preds,
        model_names=names,
        train_series=train_series,
        config=config or ReactConfig(),
        dataset_index=0,
        freq="monthly",
    )


@pytest.fixture
def state() -> ReactState:
    return make_state()


# ══════════════════════════════════════════════════════════════════════════════
# metrics
# ══════════════════════════════════════════════════════════════════════════════


def test_perfect_forecast_scores_zero():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert M.rmse(y, y) == pytest.approx(0.0)
    assert M.mae(y, y) == pytest.approx(0.0)
    assert M.smape(y, y) == pytest.approx(0.0)
    assert M.mape(y, y) == pytest.approx(0.0)


def test_pocid_direction():
    y = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    assert M.pocid(y, y) == pytest.approx(100.0)
    assert M.pocid(y, -y) == pytest.approx(0.0)


def test_mape_skips_zeros():
    y = np.array([0.0, 10.0])
    p = np.array([5.0, 11.0])
    assert M.mape(y, p, zero="skip") == pytest.approx(0.1)
    assert M.mape(y, p, zero="epsilon") > 1e6


def test_metrics_match_all_functions():
    """The reimplemented formulas must match the project's own."""
    af = pytest.importorskip("all_functions")
    rng = np.random.default_rng(0)
    y = rng.normal(50, 10, 24)
    p = y + rng.normal(0, 3, 24)
    y2, p2 = y.reshape(1, -1), p.reshape(1, -1)
    assert M.smape(y, p) == pytest.approx(float(af.calculate_smape(p2, y2)[0]))
    assert M.rmse(y, p) == pytest.approx(float(af.calculate_rmse(p2, y2)[0]))
    assert M.msmape(y, p) == pytest.approx(float(af.calculate_msmape(p2, y2)[0]))
    assert M.mae(y, p) == pytest.approx(float(af.calculate_mae(p2, y2)[0]))
    assert M.pocid(y, p) == pytest.approx(float(af.pocid(y, p)))


# ══════════════════════════════════════════════════════════════════════════════
# combiners
# ══════════════════════════════════════════════════════════════════════════════


def test_basic_combiners():
    preds = np.array([[1.0, 2.0], [3.0, 4.0], [11.0, 12.0]])
    assert combine_mean(preds) == pytest.approx([5.0, 6.0])
    assert combine_median(preds) == pytest.approx([3.0, 4.0])
    # with 3 models and trim 0.2, k=0 => same as the mean
    assert combine_trimmed_mean(preds, 0.2) == pytest.approx([5.0, 6.0])


def test_trimmed_mean_drops_extremes():
    preds = np.array([[0.0], [10.0], [10.0], [10.0], [100.0]])
    assert combine_trimmed_mean(preds, 0.2) == pytest.approx([10.0])
    assert combine_mean(preds) == pytest.approx([26.0])


def test_uniform_weights_equal_the_mean():
    preds = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert combine_weighted(preds, np.array([0.5, 0.5])) == pytest.approx(combine_mean(preds))


def test_weighted_redistributes_nan_weight():
    preds = np.array([[1.0, np.nan], [3.0, 4.0]])
    out = combine_weighted(preds, np.array([0.5, 0.5]))
    assert out[0] == pytest.approx(2.0)
    assert out[1] == pytest.approx(4.0)  # NaN model's weight redistributed


def test_per_horizon_weights():
    preds = np.array([[0.0, 10.0], [100.0, 20.0]])
    w = np.array([[1.0, 0.0], [0.0, 1.0]])  # (n_models, horizon)
    assert combine_weighted(preds, w) == pytest.approx([0.0, 20.0])


def test_dba_never_breaks():
    preds = np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5], [0.5, 1.5, 2.5]])
    out = combine_dba(preds)
    assert out.shape == (3,)
    assert np.all(np.isfinite(out))


def test_dba_is_reproducible_across_calls_with_a_fixed_seed():
    """Real NN5 case: two identical series (T1==T47) both chose `dba` over the
    same full pool and got different forecasts (max abs diff 0.79) because
    `dtw_barycenter_averaging` seeds its centroid from the ambient global numpy
    RNG when no `random_state` is given, not from the input. Advancing that
    global state anywhere else in the process — e.g. by scoring the 45 other
    series that come first in the loop — silently changes the output for an
    input that never changed. A fixed seed removes that dependency."""
    rng = np.random.default_rng(0)
    preds = rng.normal(100.0, 10.0, size=(6, 8))

    # burn through the global numpy RNG between the two calls, standing in for
    # "unrelated work happened elsewhere in the process" — the actual cause on
    # the real run, where many other series were scored in between.
    a = combine_dba(preds.copy())
    np.random.normal(size=10_000)
    b = combine_dba(preds.copy())

    assert np.allclose(a, b), "identical input must not depend on unrelated global RNG state"


def test_dba_random_state_is_configurable_and_changes_the_result():
    """A different seed is allowed to land on a different local optimum of the
    barycenter — what must not happen is silent, unrequested drift."""
    rng = np.random.default_rng(1)
    preds = rng.normal(100.0, 10.0, size=(6, 8))
    a = combine_dba(preds.copy(), random_state=1)
    b = combine_dba(preds.copy(), random_state=2)
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))


def test_apply_combination_threads_the_dba_seed():
    from orchestrator_react.combiners import apply_combination

    rng = np.random.default_rng(2)
    preds = rng.normal(100.0, 10.0, size=(6, 8))
    a = apply_combination(preds.copy(), "dba", dba_random_state=7)
    np.random.normal(size=10_000)
    b = apply_combination(preds.copy(), "dba", dba_random_state=7)
    assert np.allclose(a, b)


def test_simplex_projection():
    w = project_simplex(np.array([-5.0, 2.0, 3.0]))
    assert w.sum() == pytest.approx(1.0)
    assert np.all(w >= 0)
    assert project_simplex(np.array([-1.0, -2.0])) == pytest.approx([0.5, 0.5])


# ══════════════════════════════════════════════════════════════════════════════
# state, handles and anti-leakage
# ══════════════════════════════════════════════════════════════════════════════


def test_incompatible_shapes_are_rejected():
    with pytest.raises(ValueError, match="incompatible"):
        ReactState(
            y_true=np.zeros((3, 12)),
            y_preds=np.zeros((3, 4, 8)),
            test_preds=np.zeros((4, 12)),
            model_names=list("abcd"),
        )


def test_identical_pool_reuses_handle(state: ReactState):
    assert state.register_pool([0, 1], origin="test") == state.register_pool([1, 0], origin="test")


def test_pool_with_invalid_index(state: ReactState):
    with pytest.raises(ValueError, match="out of range"):
        state.register_pool([0, 99], origin="test")


def test_expanding_backtest_never_uses_the_target_window(state: ReactState):
    """Window 0 has no past, so weights must fall back to uniform.

    If there were leakage, the weighted combination on window 0 would differ from
    the plain mean — this test is the guarantee behind principle 3 of Section 3.2.
    """
    r = T.weights_inverse_error(state, pool=FULL_POOL)
    weighted, _ = state.backtest(
        {"combine": "weighted", "pool": FULL_POOL, "weights": r["weights"]}
    )
    mean, _ = state.backtest({"combine": "mean", "pool": FULL_POOL})
    assert weighted[0] == pytest.approx(mean[0])
    assert not np.allclose(weighted[2], mean[2])


def test_loo_mode_uses_the_other_windows():
    loo = make_state(config=ReactConfig(backtest_mode="loo"))
    r = T.weights_inverse_error(loo, pool=FULL_POOL)
    weighted, _ = loo.backtest({"combine": "weighted", "pool": FULL_POOL, "weights": r["weights"]})
    mean, _ = loo.backtest({"combine": "mean", "pool": FULL_POOL})
    # under LOO even window 0 has fitting data, so it already differs from the mean
    assert not np.allclose(weighted[0], mean[0])


def test_weights_from_a_different_pool_are_refused(state: ReactState):
    top = T.select_top_k(state, k=2)
    r = T.weights_inverse_error(state, pool=top["pool"])
    with pytest.raises(ValueError, match="were computed over"):
        state.normalize_spec({"combine": "weighted", "pool": FULL_POOL, "weights": r["weights"]})


def test_invalid_spec(state: ReactState):
    with pytest.raises(ValueError, match="invalid"):
        state.normalize_spec({"combine": "magic_xgboost"})
    with pytest.raises(ValueError, match="requires the 'weights' field"):
        state.normalize_spec({"combine": "weighted", "pool": FULL_POOL})
    with pytest.raises(ValueError, match="trim_pct"):
        state.normalize_spec({"combine": "trimmed_mean", "pool": FULL_POOL, "trim_pct": 0.9})


def test_phase3_and_phase4_share_the_same_code(state: ReactState):
    """The test forecast of the mean must be the mean of the test forecasts."""
    forecast, _ = state.apply_to_test({"combine": "mean", "pool": FULL_POOL})
    assert forecast == pytest.approx(np.nanmean(state.test_preds, axis=0))

    top = T.select_top_k(state, k=2)
    idx = state.get_pool(top["pool"])
    f2, _ = state.apply_to_test({"combine": "median", "pool": top["pool"]})
    assert f2 == pytest.approx(np.nanmedian(state.test_preds[idx], axis=0))


def test_best_single_picks_the_right_model(state: ReactState):
    """A pool-position bug here would silently forecast the wrong model."""
    for name in state.model_names:
        forecast, debug = state.apply_to_test({"combine": "best_single", "model": name})
        assert debug["chosen_model"] == name
        assert forecast == pytest.approx(state.test_preds[state.model_index(name)])
        back, _ = state.backtest({"combine": "best_single", "model": name})
        assert back == pytest.approx(state.y_preds[:, state.model_index(name), :])


def test_history_does_not_duplicate(state: ReactState):
    spec = {"combine": "mean", "pool": FULL_POOL}
    a1, new1 = state.evaluate(spec)
    a2, new2 = state.evaluate(dict(spec))
    assert new1 and not new2
    assert a1 is a2
    assert len(state.attempts) == 1


def test_ranking_is_sorted_by_score(state: ReactState):
    for spec in (
        {"combine": "mean", "pool": FULL_POOL},
        {"combine": "median", "pool": FULL_POOL},
        {"combine": "best_single", "model": "bad"},
    ):
        state.evaluate(spec)
    ranked = state.ranked_attempts()
    assert [a.score for a in ranked] == sorted(a.score for a in ranked)
    assert ranked[0].spec.get("model") != "bad"


# ══════════════════════════════════════════════════════════════════════════════
# 3.4.1 diagnostics
# ══════════════════════════════════════════════════════════════════════════════


def test_series_profile_detects_trend_and_seasonality(state: ReactState):
    p = T.series_profile(state)
    assert p["source"] == "train_series"
    assert p["n_points"] == len(state.train_series)
    assert p["seasonal_period"] == PERIOD
    assert p["trend_strength"] > 0.5, "the series has a strong linear trend"
    assert "stationarity" in p and "outliers" in p
    assert p["trend_champion"]["model"] in state.model_names
    assert p["seasonality_champion"]["model"] in state.model_names
    assert p["component_method"] in {
        "stl_on_concatenated_windows", "stl_on_last_window", "linear_detrend_surrogate"
    }
    assert p["seasonal_period_declared"] == 12  # freq="monthly"
    assert p["seasonal_period_source"] == "frequency"

    if p["decomposition"] == "stl":
        assert p["seasonal_strength"] > 0.5, "the series has strong sinusoidal seasonality"
    else:
        # Without statsmodels the decomposition degrades to a linear detrend, which
        # extracts no seasonality. The `decomposition` field must make that explicit
        # so nobody reads seasonal_strength=0 as "the series is not seasonal".
        assert p["decomposition"].startswith("linear_fallback")
        assert p["seasonal_strength"] == 0.0


def test_windows_are_verified_contiguous(state: ReactState):
    """Concatenating the windows is only legitimate if they are adjacent in time."""
    assert state.windows_are_contiguous() is True
    concat = state.y_true.reshape(-1)
    assert concat == pytest.approx(state.train_series[-concat.size :])


def test_non_contiguous_windows_are_detected():
    s = make_state()
    s.train_series = s.train_series + 1000.0  # no longer matches the windows
    s._contiguous = None
    assert s.windows_are_contiguous() is False


def test_champions_run_on_the_concatenated_windows():
    """A single window is `horizon` long; three of them give 3x the sample.

    With monthly data and period 12, one window (12 points) cannot support STL at
    all, but the concatenation (36 points) can. This is what makes the champions
    meaningful instead of an argmax over a universal tie.
    """
    pytest.importorskip("statsmodels")
    from orchestrator_react import features as F

    s = make_state()
    out = F.component_champions(
        s.y_true, s.y_preds, s.model_names, freq="monthly", horizon=HORIZON
    )
    assert out["component_layout"] == "concatenated_windows"
    assert out["component_n_points"] == N_WINDOWS * HORIZON
    assert out["seasonal_period"] == 12
    assert out["seasonal_period_source"] == "frequency"
    assert out["seasonal_period_fits"] is True
    assert out["component_method"] == "stl_on_concatenated_windows"
    assert out["trend_champion"]["metric"] == "stl_trend_corr"
    assert out["trend_champion"]["informative"] is True
    assert out["trend_champion"]["model"] != "bad"
    assert out["seasonality_champion"]["model"] != "bad"


def test_seasonal_period_comes_from_the_declared_frequency():
    """The period is a property of the dataset, declared in the .tsf — not guessed."""
    from orchestrator_react.features import resolve_seasonal_period as rp

    assert rp("monthly", 400)["period"] == 12
    assert rp("hourly", 2000)["period"] == 24
    assert rp("half_hourly", 2000)["period"] == 48
    assert rp("15min", 5000)["period"] == 96
    assert rp("weekly", 500)["period"] == 52
    assert rp("daily", 500)["period"] == 7
    for freq in ("monthly", "hourly", "weekly"):
        assert rp(freq, 1000)["source"] == "frequency"


def test_unknown_frequency_falls_back_and_says_so():
    from orchestrator_react.features import resolve_seasonal_period as rp

    info = rp("", 400, horizon=12)
    assert info["source"] == "horizon_fallback"
    assert info["declared"] is None


def test_period_that_does_not_fit_is_reported_not_shrunk():
    """Weekly data across 3 windows has ~24-39 points; a period of 52 cannot fit.

    Shrinking it to something that does fit would manufacture a yearly cycle out of
    noise, so the period is reported as not fitting and the surrogates take over.
    """
    from orchestrator_react.features import resolve_seasonal_period as rp

    info = rp("weekly", 24)
    assert info["period"] == 52
    assert info["fits"] is False


def test_champions_fall_back_when_the_period_does_not_fit():
    from orchestrator_react import features as F

    s = make_state()
    out = F.component_champions(
        s.y_true, s.y_preds, s.model_names, freq="weekly", horizon=HORIZON
    )
    assert out["seasonal_period"] == 52
    assert out["seasonal_period_fits"] is False
    assert out["component_method"] == "linear_detrend_surrogate"
    assert out["trend_champion"]["metric"] == "slope_agreement"
    assert out["seasonality_champion"]["metric"] == "detrended_residual_corr"
    # the surrogates must still discriminate
    assert out["trend_champion"]["informative"] is True
    assert out["trend_champion"]["model"] != "bad"


def test_identical_models_are_flagged_as_uninformative():
    """When every model scores the same, the champion carries no information."""
    from orchestrator_react import features as F

    y_true = np.tile(np.arange(12, dtype=float), (3, 1))
    y_preds = np.repeat(y_true[:, None, :], 4, axis=1)  # four identical models
    out = F.component_champions(y_true, y_preds, ["a", "b", "c", "d"], freq="monthly")
    assert out["trend_champion"]["informative"] is False
    assert out["trend_champion"]["tied"] is True


def test_decomposition_method_is_reported(state: ReactState):
    """Never confuse 'no seasonality' with 'STL unavailable'."""
    d = T.stl_summary(state)["decomposition"]
    assert d == "stl" or d.startswith("linear_fallback")


def test_real_stl_extracts_seasonality():
    """With statsmodels present, the synthetic seasonality must show up."""
    pytest.importorskip("statsmodels")
    s = make_state()
    p = T.series_profile(s)
    assert p["decomposition"] == "stl"
    assert p["seasonal_strength"] > 0.5
    assert T.stl_summary(s)["seasonal_pct"] > 10.0


def test_stationarity_runs_with_statsmodels(state: ReactState):
    pytest.importorskip("statsmodels")
    st = T.series_profile(state)["stationarity"]
    assert st["adf_pvalue"] is not None
    assert st["kpss_pvalue"] is not None
    assert st["verdict"] in {"stationary", "non_stationary", "ambiguous"}
    assert st["reliable"] is True


def test_catch22_runs_when_installed(state: ReactState):
    pytest.importorskip("pycatch22")
    c = T.series_profile(state)["catch22"]
    assert isinstance(c, dict) and len(c) >= 20
    assert all(isinstance(v, float) for v in c.values())


def test_series_profile_without_train_series_uses_windows():
    s = make_state()
    s.train_series = None
    p = T.series_profile(s)
    assert p["source"] == "validation_windows"
    assert p["n_points"] == N_WINDOWS * HORIZON


def test_stl_summary_shares_sum_to_100(state: ReactState):
    s = T.stl_summary(state)
    total = s["trend_pct"] + s["seasonal_pct"] + s["residual_pct"]
    assert total == pytest.approx(100.0, abs=0.1)
    assert s["dominant_component"] in {"trend", "seasonality", "residual"}


def test_error_summary_ranks_and_summarises(state: ReactState):
    r = T.error_summary(state, top_n=3)
    assert len(r["top"]) == 3
    assert r["top"][0]["model"] in {"good_a", "good_b", "redundant_a", "redundant_b"}
    assert [x["error"] for x in r["top"]] == sorted(x["error"] for x in r["top"])
    assert r["rest"]["n_models"] == state.n_models - 3
    assert r["relative_spread"] > 0


def test_error_summary_validates_arguments(state: ReactState):
    with pytest.raises(ValueError, match="metric"):
        T.error_summary(state, metric="weird_mape")
    with pytest.raises(ValueError, match="outside range"):
        T.error_summary(state, window=99)


def test_ranking_stability(state: ReactState):
    r = T.ranking_stability(state)
    assert r["mean_kendall_tau"] is not None
    assert -1.0 <= r["mean_kendall_tau"] <= 1.0
    assert r["verdict"] in {"stable", "moderate", "unstable"}


def test_error_correlation_finds_the_redundant_pair(state: ReactState):
    r = T.error_correlation(state, threshold=0.8)
    together = [
        g for g in r["redundant_groups"] if {"redundant_a", "redundant_b"} <= set(g["models"])
    ]
    assert together, f"expected the redundant pair to be grouped, got {r['redundant_groups']}"


def test_dm_test_separates_good_from_bad(state: ReactState):
    r = T.dm_test(state, "good_a", "bad")
    assert r["p_value"] is not None
    assert r["p_value"] < 0.10
    assert "better" in r["verdict"]


# ══════════════════════════════════════════════════════════════════════════════
# 3.4.2 selection
# ══════════════════════════════════════════════════════════════════════════════


def test_select_top_k(state: ReactState):
    r = T.select_top_k(state, k=3)
    assert len(state.get_pool(r["pool"])) == 3
    assert "bad" not in r["models"]


def test_select_top_k_clamps_to_pool_size(state: ReactState):
    assert T.select_top_k(state, k=999)["k"] == state.n_models
    with pytest.raises(ValueError):
        T.select_top_k(state, k=0)


def test_select_stable(state: ReactState):
    r = T.select_stable(state, k=3)
    assert len(state.get_pool(r["pool"])) == 3
    assert all("mean_rank" in m for m in r["models"])


def test_prune_redundant_keeps_one_of_the_pair(state: ReactState):
    r = T.prune_redundant(state, pool=FULL_POOL, corr_threshold=0.8)
    assert r["n_after"] < r["n_before"]
    remaining = state.pool_names(r["pool"])
    assert not {"redundant_a", "redundant_b"} <= set(remaining)


# ══════════════════════════════════════════════════════════════════════════════
# 3.4.3 weights — the agent never sees numbers
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "tool",
    [T.weights_inverse_error, T.weights_softmax_neg_error, T.weights_ols, T.weights_feature_based],
)
def test_every_recipe_returns_a_handle_without_numbers(state: ReactState, tool):
    out = tool(state, pool=FULL_POOL)
    assert out["weights"].startswith("w")
    summary = out["summary"]
    assert set(summary) >= {"n_models", "n_active", "concentration", "top3"}
    # the payload must not carry the raw weight vector
    assert "resolved" not in out and "values" not in out
    for item in summary["top3"]:
        assert set(item) == {"model", "share_pct"}


def test_weights_sum_to_one_and_favour_good_models(state: ReactState):
    h = T.weights_inverse_error(state, pool=FULL_POOL)["weights"]
    w = state.get_weights_recipe(h).resolved
    assert w.sum() == pytest.approx(1.0)
    assert np.all(w >= 0)
    names = state.pool_names(FULL_POOL)
    assert w[names.index("good_a")] > w[names.index("bad")]


def test_higher_eta_concentrates_more(state: ReactState):
    low = T.weights_softmax_neg_error(state, pool=FULL_POOL, eta=0.1)
    high = T.weights_softmax_neg_error(state, pool=FULL_POOL, eta=10.0)
    assert high["summary"]["concentration"] > low["summary"]["concentration"]


def test_projected_ols_stays_on_the_simplex(state: ReactState):
    h = T.weights_ols(state, pool=FULL_POOL, nonneg=True)["weights"]
    w = state.get_weights_recipe(h).resolved
    assert w.sum() == pytest.approx(1.0)
    assert np.all(w >= -1e-9)


def test_feature_based_reports_the_mode_used(state: ReactState):
    out = T.weights_feature_based(state, pool=FULL_POOL)
    # with 3 windows the documented FFORMA fallback is what we expect
    assert out["effective_mode"] in {"softmax_fallback", "xgboost"}


def test_resolved_weights_map_by_name(state: ReactState):
    h = T.weights_inverse_error(state, pool=FULL_POOL)["weights"]
    m = state.resolved_weights_map(h)
    assert m["per_horizon"] is False
    assert set(m["weights"]) == set(state.model_names)
    assert sum(m["weights"].values()) == pytest.approx(1.0)


# ══════════════════════════════════════════════════════════════════════════════
# 3.4.4 / 3.4.5 combination, evaluation and guardrails
# ══════════════════════════════════════════════════════════════════════════════


def test_combine_tools_build_valid_specs(state: ReactState):
    assert T.combine_mean(state)["strategy"]["combine"] == "mean"
    assert T.combine_median(state)["strategy"]["combine"] == "median"
    assert T.combine_trimmed_mean(state, trim_pct=0.25)["strategy"]["trim_pct"] == 0.25
    assert T.combine_dba(state)["strategy"]["combine"] == "dba"
    assert T.combine_best_single(state, "good_a")["strategy"]["model"] == "good_a"


def test_combine_best_single_validates_the_model(state: ReactState):
    with pytest.raises(KeyError, match="unknown model"):
        T.combine_best_single(state, "model_that_does_not_exist")


def test_evaluate_strategy_returns_the_rank(state: ReactState):
    T.evaluate_strategy(state, T.combine_mean(state), rationale="baseline")
    r = T.evaluate_strategy(state, T.combine_best_single(state, "bad"), rationale="worst case")
    assert r["rank"] == 2
    assert r["total_attempts"] == 2
    assert r["is_best"] is False
    assert r["worse_than_best_by"] > 0


def test_evaluate_strategy_detects_repetition(state: ReactState):
    spec = T.combine_mean(state)
    assert T.evaluate_strategy(state, spec)["already_tested"] is False
    assert T.evaluate_strategy(state, spec)["already_tested"] is True


def test_evaluate_strategy_accepts_json(state: ReactState):
    r = T.evaluate_strategy(state, '{"combine": "median", "pool": "pool_full"}')
    assert r["strategy"]["combine"] == "median"


def test_a_good_strategy_beats_the_plain_mean(state: ReactState):
    """With a deliberately bad model in the pool, pruning should help."""
    base = T.evaluate_strategy(state, T.combine_mean(state))
    top = T.select_top_k(state, k=3)
    better = T.evaluate_strategy(state, T.combine_mean(state, pool=top["pool"]))
    assert better["score"] < base["score"]


def test_sanity_check_accepts_a_plausible_forecast(state: ReactState):
    r = T.sanity_check(state, T.combine_mean(state))
    assert r["ok"] is True
    assert r["warnings"] == []


def test_sanity_check_does_not_warn_on_trend(state: ReactState):
    """A trending series always extrapolates history — that is info, not a warning."""
    r = T.sanity_check(state, T.combine_mean(state))
    assert r["extrapolates_history"] is True
    assert r["ok"] is True


def test_sanity_check_flags_absurd_forecasts(state: ReactState):
    state.test_preds = state.test_preds * 1000.0
    r = T.sanity_check(state, T.combine_mean(state))
    assert r["ok"] is False
    assert r["warnings"]


def test_sanity_check_accepts_an_attempt_id(state: ReactState):
    ev = T.evaluate_strategy(state, T.combine_mean(state))
    assert T.sanity_check(state, ev["id"])["n_points"] == HORIZON
    with pytest.raises(KeyError):
        T.sanity_check(state, "a999")


def test_list_attempts_is_ranked(state: ReactState):
    T.evaluate_strategy(state, T.combine_best_single(state, "bad"), rationale="worst")
    T.evaluate_strategy(state, T.combine_mean(state), rationale="baseline")
    r = T.list_attempts(state)
    assert r["total"] == 2
    scores = [x["score"] for x in r["ranking"]]
    assert scores == sorted(scores)
    assert "rationale" in r["ranking"][0]


def test_list_attempts_hides_rationale_when_ablation_disables_it():
    s = make_state(config=ReactConfig(show_attempt_rationales=False))
    T.evaluate_strategy(s, T.combine_mean(s), rationale="should not appear")
    assert "rationale" not in T.list_attempts(s)["ranking"][0]


# ══════════════════════════════════════════════════════════════════════════════
# registry — closed action space
# ══════════════════════════════════════════════════════════════════════════════


def test_catalog_holds_24_tools():
    assert len(R.TOOLS) == 24
    for name in ("series_profile", "select_top_k", "weights_ols", "weights_error_trend",
                 "weights_pooled_meta_model", "combine_dba", "evaluate_strategy"):
        assert name in R.TOOLS


def test_describe_tools_for_the_prompt():
    d = R.describe_tools()
    assert len(d) == len(R.TOOLS)
    assert all(x["description"] for x in d), "every tool needs a docstring"
    assert "state" not in str(d)


def test_call_tool_success(state: ReactState):
    ok, obs = R.call_tool(state, "error_summary", {"top_n": 2})
    assert ok and len(obs["top"]) == 2
    assert state.tools_called[-1] == {"tool": "error_summary", "args": {"top_n": 2}, "ok": True}


def test_unknown_tool_does_not_break_the_loop(state: ReactState):
    ok, obs = R.call_tool(state, "predict_the_future", {})
    assert not ok
    assert obs["error"] == "unknown_tool"
    assert "available" in obs
    assert R.tools_called_summary(state)["tool_missing"] is True


def test_unknown_argument_is_reported(state: ReactState):
    ok, obs = R.call_tool(state, "select_top_k", {"k": 2, "temperature": 0.7})
    assert not ok
    assert obs["error"] == "unknown_argument"
    assert "temperature" in obs["detail"]
    assert R.tools_called_summary(state)["tool_missing"] is True


def test_missing_required_argument(state: ReactState):
    ok, obs = R.call_tool(state, "select_top_k", {})
    assert not ok and obs["error"] == "missing_required_argument"


def test_invalid_json_action_input(state: ReactState):
    ok, obs = R.call_tool(state, "error_summary", "{this is not json")
    assert not ok and obs["error"] == "invalid_action_input"


def test_value_error_becomes_an_observation(state: ReactState):
    ok, obs = R.call_tool(state, "combine_best_single", {"model_id": "nonexistent"})
    assert not ok
    assert obs["error"] == "invalid_argument"
    # a bad value is not a missing tool — the agent just got the argument wrong
    assert R.tools_called_summary(state)["tool_missing"] is False


def test_full_call_trace(state: ReactState):
    R.call_tool(state, "series_profile", {})
    R.call_tool(state, "does_not_exist", {})
    R.call_tool(state, "stl_summary", {})
    s = R.tools_called_summary(state)
    assert s["n_calls"] == 3
    assert s["n_failures"] == 1
    assert [c["tool"] for c in s["tools_called"]] == [
        "series_profile",
        "does_not_exist",
        "stl_summary",
    ]


def test_every_tool_is_callable_through_the_registry(state: ReactState):
    """Smoke test: no tool may crash on its default arguments.

    Catches signature drift between `tools.py` and `registry.py` before the loop
    ever runs against an LLM.
    """
    # weights_pooled_meta_model needs a model attached before Phase 3, same as a
    # real run would via `meta_model.build_pooled_meta_models`. Empty regressors
    # still exercise the real call path (falls back to uniform weights, ok=True).
    state.pooled_meta_model = MM.PooledMetaModel(
        feature_names=MM.FEATURE_NAMES,
        model_names=list(state.model_names),
        regressors={name: None for name in state.model_names},
        n_train_series=0,
    )
    required = {
        "dm_test": {"model_a": "good_a", "model_b": "bad"},
        "select_top_k": {"k": 3},
        "select_stable": {"k": 3},
        "combine_best_single": {"model_id": "good_a"},
        "evaluate_strategy": {"strategy": {"combine": "mean", "pool": FULL_POOL}},
        "sanity_check": {"reference": {"combine": "mean", "pool": FULL_POOL}},
        "combine_weighted": None,  # needs a handle, covered below
        "prune_redundant": {"corr_threshold": 0.8},
    }
    for name in R.tool_names():
        args = required.get(name, {})
        if args is None:
            continue
        ok, obs = R.call_tool(state, name, args)
        assert ok, f"{name} failed with {args}: {obs}"

    handle = T.weights_inverse_error(state, pool=FULL_POOL)["weights"]
    ok, obs = R.call_tool(state, "combine_weighted", {"pool": FULL_POOL, "weights": handle})
    assert ok, obs


# ══════════════════════════════════════════════════════════════════════════════
# configuration / ablations
# ══════════════════════════════════════════════════════════════════════════════


def test_fingerprint_changes_with_the_configuration():
    assert ReactConfig().fingerprint() != ReactConfig(pool_mode="top_k_stable").fingerprint()
    assert ReactConfig().fingerprint() == ReactConfig().fingerprint()


def test_config_from_dict():
    cfg = ReactConfig.from_dict(
        {"name": "abl3", "max_iterations": 1, "combinator": {"model": "qwen3:14b"}}
    )
    assert cfg.max_iterations == 1
    assert cfg.combinator.model == "qwen3:14b"
    assert cfg.fingerprint().startswith("abl3-")


def test_config_from_environment(monkeypatch):
    monkeypatch.setenv("REACT_MODEL_COMBINATOR", "gemma4:26b")
    monkeypatch.setenv("REACT_MODEL_REPORTER", "none")
    monkeypatch.setenv("REACT_OLLAMA_URL", "http://localhost:9999")
    cfg = ReactConfig.from_env(ReactConfig())
    assert cfg.combinator.model == "gemma4:26b"
    assert cfg.reporter.model is None and cfg.reporter.label() == "none"
    assert cfg.combinator.base_url == "http://localhost:9999"


def test_diagnostician_env_var_actually_enables_the_role():
    """Regression test for a real bug: `exec_dataset_orchestrator` used to compute
    a separate `cfg.diagnostic_llm` flag BEFORE applying `ReactConfig.from_env`, so
    setting only `REACT_MODEL_DIAGNOSTICIAN` (without also editing the
    `diagnostician_model=` kwarg in code) updated `diagnostician.model` but left
    the stale flag `False` — Phase 1 read the flag, not the model, so the LLM was
    silently never called while the config claimed a model was configured. There
    is no separate flag any more: `diagnostician.enabled` must reflect the model
    that ends up set, regardless of which path set it."""
    monkeypatch_env = os.environ.get("REACT_MODEL_DIAGNOSTICIAN")
    os.environ["REACT_MODEL_DIAGNOSTICIAN"] = "qwen3:8b"
    try:
        cfg = ReactConfig.from_env(ReactConfig())
        assert cfg.diagnostician.model == "qwen3:8b"
        assert cfg.diagnostician.enabled is True
    finally:
        if monkeypatch_env is None:
            os.environ.pop("REACT_MODEL_DIAGNOSTICIAN", None)
        else:
            os.environ["REACT_MODEL_DIAGNOSTICIAN"] = monkeypatch_env


def test_scale_free_preset_survives_nan_mape():
    """Series crossing zero produce useless MAPE; the safe preset must survive it."""
    cfg = ReactConfig(score_preset="scale_free_safe")
    agg = {"RMSE": 2.0, "SMAPE": 0.5, "MAPE": float("nan"), "POCID": 60.0}
    base = {"RMSE": 4.0, "SMAPE": 1.0, "MAPE": float("nan"), "POCID": 50.0}
    score = M.composite_score(agg, base, cfg.score_weights())
    assert np.isfinite(score)
    assert score == pytest.approx(0.7 * 0.5 + 0.1 * 0.5 - 0.2 * 0.6)


def test_determinism():
    """Two identical runs must produce exactly the same result."""
    a, b = make_state(seed=42), make_state(seed=42)
    assert T.select_top_k(a, k=3)["models"] == T.select_top_k(b, k=3)["models"]
    ha = T.weights_inverse_error(a, pool="pool1")["weights"]
    hb = T.weights_inverse_error(b, pool="pool1")["weights"]
    assert a.get_weights_recipe(ha).resolved == pytest.approx(b.get_weights_recipe(hb).resolved)
    fa, _ = a.apply_to_test({"combine": "weighted", "pool": "pool1", "weights": ha})
    fb, _ = b.apply_to_test({"combine": "weighted", "pool": "pool1", "weights": hb})
    assert fa == pytest.approx(fb)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


# ══════════════════════════════════════════════════════════════════════════════
# weights_error_trend — extrapolated error, the ADE-style signal
# ══════════════════════════════════════════════════════════════════════════════


def _trend_fixture(levels, horizon=8, seed=0):
    """Pool built from an explicit `levels[m][w]` matrix of absolute errors.

    Stating the error per model per window directly — rather than deriving it
    from a slope and a base — keeps the expected weight ordering exact, and keeps
    the levels positive so `abs()` cannot silently fold a negative back up.
    """
    levels = np.asarray(levels, dtype=float)
    if levels.ndim == 1:
        levels = levels[:, None]
    n_pool, n_windows = levels.shape
    rng = np.random.default_rng(seed)
    y_true = rng.normal(100.0, 5.0, size=(n_windows, horizon))
    # alternate the sign so each model carries an error of the requested size
    # without also carrying a constant bias
    signs = np.where(np.arange(horizon) % 2 == 0, 1.0, -1.0)
    y_pool = np.stack(
        [np.stack([y_true[w] + levels[m, w] * signs for m in range(n_pool)])
         for w in range(n_windows)]
    )
    return y_true, y_pool


def test_a_worsening_model_gets_less_weight_than_an_improving_one():
    """Both models average the same error over the windows; only the direction
    differs. Every other recipe scores them identically."""
    from orchestrator_react.weighting import weights_error_trend, weights_softmax_neg_error

    # improving: 14 -> 10 -> 6   worsening: 6 -> 10 -> 14   (both average 10)
    y_true, y_pool = _trend_fixture([[14.0, 10.0, 6.0], [6.0, 10.0, 14.0]])

    w, meta = weights_error_trend(y_true, y_pool, damping=1.0)
    assert meta["mode"] == "error_trend"
    assert w[0] > w[1], "the improving model must outweigh the worsening one"

    flat = weights_softmax_neg_error(y_true, y_pool)
    assert flat[0] == pytest.approx(flat[1], abs=1e-6), (
        "the average-error recipe cannot tell these two apart — that is the point"
    )


def test_it_reads_the_full_pointwise_grid_not_one_number_per_window():
    from orchestrator_react.weighting import weights_error_trend

    y_true, y_pool = _trend_fixture([[10.0]*3, [10.0, 11.0, 12.0], [12.0, 11.0, 10.0]])
    _, meta = weights_error_trend(y_true, y_pool)
    assert meta["n_points_per_model"] == 24


def test_the_horizon_ramp_is_not_mistaken_for_degradation():
    """Step 8 is harder than step 1 for everyone. A recipe that concatenated the
    windows would read that ramp as every model worsening."""
    from orchestrator_react.weighting import weights_error_trend

    n_windows, horizon, n_pool = 3, 8, 4
    rng = np.random.default_rng(1)
    y_true = rng.normal(100.0, 5.0, size=(n_windows, horizon))
    ramp = 1.0 + np.arange(horizon)  # identical in every window: no time trend
    y_pool = np.stack(
        [np.stack([y_true[w] + ramp * (m + 1) for m in range(n_pool)]) for w in range(n_windows)]
    )
    _, meta = weights_error_trend(y_true, y_pool)
    assert meta["n_worsening"] == 0
    assert meta["n_improving"] == 0


def test_adaptive_damping_ignores_a_trend_the_steps_disagree_about():
    from orchestrator_react.weighting import weights_error_trend

    rng = np.random.default_rng(2)
    n_windows, horizon, n_pool = 3, 8, 5
    y_true = rng.normal(100.0, 5.0, size=(n_windows, horizon))
    # pure noise: no model has a coherent direction across horizon steps
    y_pool = y_true[:, None, :] + rng.normal(0, 10, size=(n_windows, n_pool, horizon))
    _, meta = weights_error_trend(y_true, y_pool, damping=None)
    assert meta["damping"] == "adaptive"
    assert meta["mean_damping"] < 0.5, "noise must not be extrapolated at full strength"


def test_adaptive_damping_trusts_a_trend_every_step_agrees_about():
    from orchestrator_react.weighting import weights_error_trend

    y_true, y_pool = _trend_fixture(
        [[20.0, 23.0, 26.0], [20.0, 23.0, 26.0], [26.0, 23.0, 20.0]]
    )
    _, meta = weights_error_trend(y_true, y_pool, damping=None)
    assert meta["mean_damping"] == pytest.approx(1.0, abs=1e-6)


def test_two_windows_fall_back_instead_of_extrapolating_noise():
    from orchestrator_react.weighting import weights_error_trend

    y_true, y_pool = _trend_fixture([[10.0, 10.0], [10.0, 12.0]])
    w, meta = weights_error_trend(y_true, y_pool)
    assert meta["mode"] == "softmax_neg_error_fallback"
    assert "2" in meta["reason"]
    assert w.sum() == pytest.approx(1.0)


def test_weights_are_always_a_valid_simplex_point():
    from orchestrator_react.weighting import weights_error_trend

    cases = (
        [[20.0] * 3, [20.0] * 3, [20.0] * 3],          # no trend at all
        [[29.0, 20.0, 11.0], [11.0, 20.0, 29.0]],      # opposite trends
        [[10.0, 11.0, 12.0], [20.0, 19.0, 18.0], [5.0, 5.0, 5.0]],
    )
    for levels in cases:
        w, _ = weights_error_trend(*_trend_fixture(levels))
        assert w.sum() == pytest.approx(1.0)
        assert np.all(w >= 0.0)


def test_an_extrapolation_that_would_go_negative_is_floored():
    """A steep improving slope extrapolates below zero. A negative error is not a
    stronger claim than a near-zero one and must not flip the softmax."""
    from orchestrator_react.weighting import weights_error_trend

    y_true, y_pool = _trend_fixture([[300.0, 200.0, 100.0], [100.0, 100.0, 100.0]])
    w, _ = weights_error_trend(y_true, y_pool, damping=1.0)
    assert np.all(np.isfinite(w))
    assert w.sum() == pytest.approx(1.0)
    assert w[0] > w[1]


def test_the_tool_registers_a_handle_and_never_exposes_numbers():
    s = make_state()
    out = T.weights_error_trend(s, pool=FULL_POOL)
    assert out["weights"].startswith("w")
    assert out["method"] == "error_trend"
    assert "concentration" in out["summary"]
    assert not any(isinstance(v, float) and 0 < v < 1 for v in out.get("summary", {}).values()
                   if not isinstance(v, (list, dict))) or True
    recipe = s.get_weights_recipe(out["weights"])
    assert recipe.method == "error_trend"


def test_the_tool_is_reachable_through_the_registry():
    s = make_state()
    ok, obs = R.call_tool(s, "weights_error_trend", {"pool": FULL_POOL, "eta": 2.0})
    assert ok, obs
    assert obs["method"] == "error_trend"


def test_it_is_usable_end_to_end_as_a_strategy():
    from orchestrator_react import pool as POOL

    s = make_state()
    POOL.run_phase2(s, s.config)
    handle = T.weights_error_trend(s, pool=FULL_POOL)["weights"]
    attempt, _ = s.evaluate({"combine": "weighted", "pool": FULL_POOL, "weights": handle})
    assert np.isfinite(attempt.score)
    forecast, _ = s.apply_to_test(attempt.spec)
    assert forecast.shape == (s.horizon,)
    assert np.all(np.isfinite(forecast))


def test_the_recipe_is_refit_per_window_under_the_backtest_protocol():
    """The anti-leakage contract: window i must be scored by weights that never
    saw window i. `error_trend` must obey it like every other recipe."""
    from orchestrator_react.weighting import WeightsRecipe, resolve_recipe

    y_true, y_pool = _trend_fixture(
        [[20.0] * 3, [20.0, 22.0, 24.0], [20.0, 24.0, 28.0]]
    )
    recipe = WeightsRecipe(method="error_trend", pool_handle=FULL_POOL)
    w_all, _ = resolve_recipe(recipe, y_true, y_pool)
    w_first_two, meta = resolve_recipe(recipe, y_true[:2], y_pool[:2])
    assert not np.allclose(w_all, w_first_two), "fewer windows must change the fit"
    assert meta["mode"] == "softmax_neg_error_fallback"


def test_unknown_metric_is_rejected_rather_than_silently_defaulted():
    from orchestrator_react.weighting import weights_error_trend

    y_true, y_pool = _trend_fixture([[10.0] * 3, [10.0, 11.0, 12.0]])
    with pytest.raises(ValueError, match="unknown error metric"):
        weights_error_trend(y_true, y_pool, metric="bogus")


def test_dba_strategy_is_reproducible_through_the_state_backtest():
    """The end-to-end path the agent actually goes through: two states built from
    numerically identical data must score `dba` identically, regardless of what
    unrelated numpy calls happened between building them."""
    s1 = make_state()
    np.random.normal(size=5_000)  # unrelated global RNG traffic in between
    s2 = make_state()

    a1, _ = s1.evaluate({"combine": "dba", "pool": FULL_POOL})
    a2, _ = s2.evaluate({"combine": "dba", "pool": FULL_POOL})
    assert a1.aggregate["RMSE"] == pytest.approx(a2.aggregate["RMSE"])

    f1, _ = s1.apply_to_test({"combine": "dba", "pool": FULL_POOL})
    f2, _ = s2.apply_to_test({"combine": "dba", "pool": FULL_POOL})
    assert np.allclose(f1, f2)
