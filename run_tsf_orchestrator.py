import os
import re
import json
import pandas as pd
import numpy as np
from streamfuels.datasets import DatasetLoader
from sklearn.metrics import mean_absolute_percentage_error as mape
from orchestrator_langchain.context import read_model_preds
from all_functions import calculate_smape, calculate_rmse, calculate_msmape, calculate_mae, pocid


def extract_values(list_str):
    if isinstance(list_str, str):
        numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", list_str)
        return [float(num) for num in numbers]
    return []


# def read_model_preds(model_name, dataset_index, dataset="ANP_MONTHLY"):
#     df = pd.read_csv(
#         f"./timeseries/mestrado/resultados/{model_name}/normal/{dataset}.csv",
#         sep=";",
#     )
#     df = df[df["dataset_index"] == dataset_index]

#     df["start_test"] = pd.to_datetime(df["start_test"], errors="coerce", infer_datetime_format=True)
#     df["final_test"] = pd.to_datetime(df["final_test"], errors="coerce", infer_datetime_format=True)
#     df = df.sort_values(by="start_test")

#     return df


# Columns common to every pipeline version (metrics + final-combination traceability).
COLS_BASE = [
    "dataset_index",
    "horizon",
    "regressor",
    "mape",
    "pocid",
    "smape",
    "rmse",
    "msmape",
    "mae",
    "test",
    "predictions",
    "start_test",
    "final_test",
    "description",
    "llm_artifacts_path",
    "score_preset",
    "tool_missing",
    "tools_called",
    "final_candidate_names",
    "final_candidate_count",
    # Traceability of the final combination applied on final_test
    "best_strategy_name",
    "best_strategy_method",
    "best_strategy_params",
    "predict_debug",
    "selected_base_models",
    "weights_by_horizon",
]

# Legacy columns: V1 (PatternAnalyst + Proposer/Skeptic/Statistician + debate) and
# V2 (SeriesAnnotator + StrategySelector + oracle/fixed baselines). Only written when
# running those versions — V3 does not populate them, so they are excluded from V3 output.
COLS_LEGACY = [
    # V1: debate + role outputs
    "debate_ran",
    "debate_trigger",
    "approach_pre_debate",
    "approach_post_debate",
    "debate_explanation",
    "selection_explanation",
    "when_good",
    "decision_report",
    "proposer_selected_names",
    "proposer_params_overrides",
    "proposer_force_debate",
    "proposer_debate_margin",
    "skeptic_remove_names",
    "skeptic_add_names",
    "skeptic_params_overrides",
    "statistician_remove_names",
    "statistician_add_names",
    "statistician_params_overrides",
    "proposer_think",
    "skeptic_think",
    "statistician_think",
    "pattern_analyst_think",
    "pattern_analyst_trend_champion",
    "pattern_analyst_seas_champion",
    "pattern_analyst_method_hint",
    "pattern_analyst_narrative",
    # V2: oracle comparison
    "oracle_best_name",
    "oracle_best_score",
    "oracle_best_method",
    "oracle_n_candidates",
    "llm_vs_oracle_delta",
    "llm_in_oracle_top5",
    # V2: fixed baselines
    "baseline_equal_weights_score",
    "baseline_best_single_score",
    "baseline_best_single_model",
    "llm_vs_equal_weights_delta",
    "llm_vs_best_single_delta",
    # V2: structured annotations
    "series_profile",
    "strategy_reasoning",
    "series_annotator_think",
    "strategy_selector_think",
]

# V3 columns: SeriesAnalyst → ModelCritic (prune) → CombinationArchitect + DM gate.
COLS_V3 = [
    "series_profile",
    "survivors",
    "pruned_models",
    "prune_blocked_by_mcs",
    "mcs_superior_set",
    "regime",
    "shrinkage_lambda",
    "fellback_to_pruned_mean",
    "oracle_regime",
    "llm_picked_best_regime",
    # V3 baseline scores + deltas (publication evidence: negative delta = better than baseline)
    "full_mean_score",
    "full_median_score",
    "pruned_equal_weights_score",
    "llm_regime_score",
    "chosen_score",
    "delta_chosen_vs_full_mean",
    "delta_chosen_vs_full_median",
    "delta_chosen_vs_pruned_mean",
    "delta_pruned_mean_vs_full_mean",
    # Sprint-1 additions: dual anchor (mean/median) + upstream pool curation
    "pruned_mean_score",
    "pruned_median_score",
    "anchor_choice",
    "delta_chosen_vs_pruned_median",
    "pool_curated_size",
    "pool_curated_removed",
    # V3 think blocks
    "series_analyst_think",
    "model_critic_think",
    "combination_architect_think",
]


def cols_for_version(version: str):
    """Active CSV schema for a pipeline version: base + (V3 | legacy V1/V2)."""
    if str(version).startswith("v3"):
        return COLS_BASE + COLS_V3
    return COLS_BASE + COLS_LEGACY


def _extract_think_blocks(text: str) -> str:
    """Extract concatenated <think>...</think> blocks from raw model output."""

    if not isinstance(text, str) or not text:
        return ""
    out = []
    start = 0
    while True:
        s = text.find("<think>", start)
        if s == -1:
            break
        e = text.find("</think>", s)
        if e == -1:
            break
        out.append(text[s + len("<think>") : e].strip())
        start = e + len("</think>")
    return "\n\n".join([x for x in out if x])


def get_predictions_models(models, dataset_index, final_test, dataset="ANP_MONTHLY"):
    final_test_predictions = {}
    final_test_data = None

    final_test_date = pd.to_datetime(final_test, errors="coerce")

    for model in models:
        df = read_model_preds(model, dataset_index, dataset=dataset)
        test_df = df[df["final_test"] == final_test_date]

        if not test_df.empty:
            final_row = test_df.iloc[0]
            final_test_predictions[model] = extract_values(final_row["predictions"])
            final_test_data = extract_values(final_row["test"])

    return final_test_predictions, final_test_data

import orchestrator.utils as _utils
def exec_dataset_orchestrator(
    models,
    dataset,
    use_llm: bool = False,
    proposer_model: _utils.ModelConfig = None,
    skeptic_model: _utils.ModelConfig = None,
    statistician_model: _utils.ModelConfig = None,
    pattern_analyst_model: _utils.ModelConfig = None,
    # V2 models
    series_annotator_model: _utils.ModelConfig = None,
    strategy_selector_model: _utils.ModelConfig = None,
    # V3 models
    series_analyst_model: _utils.ModelConfig = None,
    model_critic_model: _utils.ModelConfig = None,
    combination_architect_model: _utils.ModelConfig = None,
    debug: bool = False,
    rolling: str = "expanding",
    train_window: int = 3,
    llm_logs: bool = True,
    # start_index: int = 0,
    # end_index: int = 182,
    version: str = "v1_pattern",
    # Explicit path to the original .tsf (preferred). The full leakage-safe history
    # (series[:-horizon]) is read from here for the SeriesAnalyst features. If None, it is
    # resolved case-insensitively from the dataset name; if still not found, the pipeline
    # falls back to the validation-window proxy.
    original_tsf_path: str = None,
):
    # dataset = "ANP_MONTHLY"
    # dataset = "ETTH1"
    dataset_file = f"./timeseries/mestrado/resultados/catboost/normal/{dataset}.csv"
    df_dt = pd.read_csv(dataset_file, sep=";")


    df_dt["final_test"] = pd.to_datetime(
        df_dt["final_test"],
        errors="coerce",          # transforma inválidos em NaT
    )
    
    if df_dt["final_test"].isna().sum() > 0:
        print("Existem datas inválidas que viraram NaT.")

    df_new_dt = df_dt.sort_values("final_test").reset_index(drop=True)
    # df_new_dt.iloc[-1]['horizon']
    exp_name = f"orchestrator_llm_{version}" if use_llm else f"orchestrator_deterministic_{version}"
    horizon =df_new_dt.iloc[-1]['horizon']
    final_test = df_new_dt.iloc[-1]['final_test']
    num_series = df_dt["dataset_index"].nunique()

    path_experiments = f"./timeseries/mestrado/resultados/{exp_name}/"
    path_csv = f"{path_experiments}/{dataset}.csv"
    path_llm_artifacts = f"{path_experiments}/llm_artifacts/{dataset}/"
    os.makedirs(path_experiments, exist_ok=True)
    os.makedirs(path_llm_artifacts, exist_ok=True)

    from orchestrator_langchain.context import CONTEXT_MEMORY, generate_all_validations_context, init_context
    from orchestrator.pipeline import run_deterministic_pipeline, run_llm_pipeline
    from orchestrator_langchain.pipeline import run_langchain_pipeline, run_langchain_pipeline_v2, run_langchain_pipeline_v3

    # Active CSV schema for this version (V3 drops the legacy V1/V2 columns).
    cols_serie = cols_for_version(version)

    # Ensure CSV schema is up-to-date (add missing columns if file already exists).
    if not os.path.exists(path_csv):
        pd.DataFrame(columns=cols_serie).to_csv(path_csv, sep=";", index=False)
    else:
        try:
            df_existing = pd.read_csv(path_csv, sep=";")
            missing = [c for c in cols_serie if c not in df_existing.columns]
            if missing:
                for c in missing:
                    df_existing[c] = np.nan
                df_existing = df_existing.reindex(columns=cols_serie)
                df_existing.to_csv(path_csv, sep=";", index=False)
        except Exception:
            # If the existing file is malformed, keep running; new rows will still append.
            pass

    for i in range(num_series):
        init_context()
        CONTEXT_MEMORY["models_available"] = models
        generate_all_validations_context(models, i, train_window=train_window, dataset=dataset, tsf_path=original_tsf_path)
        print(f"----- DATASET INDEX: {i} -----")
        if use_llm:
            try:
                _is_v3 = version.startswith("v3")
                _is_v2 = version.startswith("v2")
                if _is_v3:
                    result = run_langchain_pipeline_v3(
                        series_analyst_model=series_analyst_model,
                        model_critic_model=model_critic_model,
                        combination_architect_model=combination_architect_model,
                        debug=debug,
                        rolling_mode=rolling,
                        train_window=train_window,
                        require_tool_call=True,
                        llm_logs=llm_logs,
                    )
                elif _is_v2:
                    result = run_langchain_pipeline_v2(
                        series_annotator_model=series_annotator_model,
                        strategy_selector_model=strategy_selector_model,
                        debug=debug,
                        rolling_mode=rolling,
                        train_window=train_window,
                        require_tool_call=True,
                        llm_logs=llm_logs,
                    )
                else:
                    result = run_langchain_pipeline(
                        proposer_model,
                        skeptic_model,
                        statistician_model,
                        pattern_analyst_model,
                        debug=debug,
                        rolling_mode=rolling,
                        train_window=train_window,
                        require_tool_call=True,
                        llm_logs=llm_logs,
                    )
            except Exception as e:
                tools_called = None
                try:
                    tools_called = list(CONTEXT_MEMORY.get("tools_called", []))
                except Exception:
                    tools_called = None

                llm_artifacts_path = ""
                try:
                    artifacts = CONTEXT_MEMORY.get("orchestrator_llm_artifacts")
                    llm_artifacts_path = os.path.abspath(os.path.join(path_llm_artifacts, f"dataset_{i}.json"))
                    payload = {
                        "dataset_index": i,
                        "exception": {
                            "type": type(e).__name__,
                            "message": str(e),
                        },
                        "tools_called": tools_called,
                        "artifacts": artifacts if isinstance(artifacts, dict) else None,
                        "context_snapshot": None,
                    }
                    try:
                        payload["context_snapshot"] = dict(CONTEXT_MEMORY)
                    except Exception:
                        payload["context_snapshot"] = "unavailable"

                    with open(llm_artifacts_path, "w", encoding="utf-8") as f:
                        json.dump(payload, f, ensure_ascii=False, indent=2)
                except Exception:
                    llm_artifacts_path = ""

                result = {
                    "success": False,
                    "description": {
                        "mode": "llm",
                        "error": "LLM pipeline failed (hard-stop)",
                        "dataset_index": i,
                        "exception": str(e),
                        "exception_type": type(e).__name__,
                        "tools_called": tools_called,
                        "llm_artifacts_path": llm_artifacts_path,
                    },
                    "result": [],
                    "debate": {
                        "debate_ran": False,
                        "debate_trigger": "exception",
                    },
                }
        else:
            result = run_deterministic_pipeline()

        _, test = get_predictions_models(models, dataset_index=i, final_test=final_test, dataset=dataset.upper())

        description = result.get("description", "")
        if not isinstance(description, str):
            try:
                description = json.dumps(description, ensure_ascii=False)
            except Exception:
                description = str(description)
        preds_real = result.get("result", [])

        debate_explanation = ""
        selection_explanation = ""
        when_good = ""
        decision_report = ""
        llm_artifacts_path = ""

        score_preset = ""
        tool_missing = np.nan
        tools_called_csv = ""
        proposer_selected_names = ""
        proposer_params_overrides = ""
        proposer_force_debate = np.nan
        proposer_debate_margin = np.nan
        final_candidate_names = ""
        final_candidate_count = np.nan
        skeptic_remove_names = ""
        skeptic_add_names = ""
        skeptic_params_overrides = ""
        statistician_remove_names = ""
        statistician_add_names = ""
        statistician_params_overrides = ""

        proposer_think = ""
        skeptic_think = ""
        statistician_think = ""
        pattern_analyst_think = ""
        pattern_analyst_trend_champion = ""
        pattern_analyst_seas_champion = ""
        pattern_analyst_method_hint = ""
        pattern_analyst_narrative = ""

        best_strategy_name = ""
        best_strategy_method = ""
        best_strategy_params = ""
        predict_debug_csv = ""
        selected_base_models = ""
        weights_by_horizon = ""

        # V2 oracle + annotation fields
        oracle_best_name = ""
        oracle_best_score = np.nan
        oracle_best_method = ""
        oracle_n_candidates = np.nan
        llm_vs_oracle_delta = np.nan
        llm_in_oracle_top5 = np.nan
        series_profile_csv = ""
        strategy_reasoning_csv = ""
        series_annotator_think = ""
        strategy_selector_think = ""

        # V2 fixed baselines
        baseline_equal_weights_score = np.nan
        baseline_best_single_score = np.nan
        baseline_best_single_model = ""
        llm_vs_equal_weights_delta = np.nan
        llm_vs_best_single_delta = np.nan

        # V3 pruning + robust combination
        survivors_csv = ""
        pruned_models_csv = ""
        prune_blocked_by_mcs_csv = ""
        mcs_superior_set_csv = ""
        regime_v3 = ""
        shrinkage_lambda_v3 = np.nan
        fellback_to_pruned_mean = np.nan
        oracle_regime_v3 = ""
        llm_picked_best_regime = np.nan
        full_mean_score = np.nan
        full_median_score = np.nan
        pruned_equal_weights_score = np.nan
        llm_regime_score = np.nan
        chosen_score_v3 = np.nan
        delta_chosen_vs_full_mean = np.nan
        delta_chosen_vs_full_median = np.nan
        delta_chosen_vs_pruned_mean = np.nan
        delta_pruned_mean_vs_full_mean = np.nan
        # Sprint-1
        pruned_mean_score = np.nan
        pruned_median_score = np.nan
        anchor_choice = ""
        delta_chosen_vs_pruned_median = np.nan
        pool_curated_size = np.nan
        pool_curated_removed_csv = ""
        series_analyst_think = ""
        model_critic_think = ""
        combination_architect_think = ""

        debate_ran = np.nan
        debate_trigger = np.nan
        approach_pre = np.nan
        approach_post = np.nan
        if use_llm:
            debate_info = result.get("debate") if isinstance(result, dict) else None
            if isinstance(debate_info, dict):
                debate_ran = bool(debate_info.get("debate_ran", False))
                debate_trigger = debate_info.get("debate_trigger")
                pre = debate_info.get("best_pre_debate")
                post = debate_info.get("best_post_debate")
                if isinstance(pre, dict):
                    approach_pre = pre.get("name")
                if isinstance(post, dict):
                    approach_post = post.get("name")
                # If debate didn't run, keep both as the final best.
                if not debate_ran:
                    best_now = result.get("best")
                    if isinstance(best_now, dict):
                        approach_pre = best_now.get("name")
                        approach_post = best_now.get("name")

            expl = result.get("explanations") if isinstance(result, dict) else None
            if isinstance(expl, dict):
                # Prefer orchestrator text; fall back to skeptic/statistician when available.
                debate_explanation = str(expl.get("orchestrator_debate_notes") or expl.get("skeptic_rationale") or "")
                selection_explanation = str(expl.get("orchestrator_reasoning") or "")
                when_good = str(expl.get("orchestrator_when_good") or expl.get("statistician_when_good") or expl.get("skeptic_when_good") or "")

            try:
                decision_report = (
                    f"pre={approach_pre} | post={approach_post} | score_preset={score_preset} | debate_ran={debate_ran} | trigger={debate_trigger} | "
                    f"debate_note={debate_explanation} | selection={selection_explanation} | when_good={when_good}"
                )
            except Exception:
                decision_report = ""

            # Persist full LLM artifacts for auditability (raw prompts/outputs).
            artifacts = result.get("llm_artifacts") if isinstance(result, dict) else None
            if isinstance(artifacts, dict):
                try:
                    llm_artifacts_path = os.path.abspath(os.path.join(path_llm_artifacts, f"dataset_{i}.json"))
                    with open(llm_artifacts_path, "w", encoding="utf-8") as f:
                        json.dump({"dataset_index": i, "artifacts": artifacts}, f, ensure_ascii=False, indent=2)
                except Exception:
                    llm_artifacts_path = ""
                try:
                    raw = artifacts.get("raw", {}) if isinstance(artifacts.get("raw"), dict) else {}
                    proposer_think = _extract_think_blocks(str(raw.get("proposer", "")))
                    skeptic_think = _extract_think_blocks(str(raw.get("skeptic", "")))
                    statistician_think = _extract_think_blocks(str(raw.get("statistician", "")))
                    pattern_analyst_think = _extract_think_blocks(str(raw.get("pattern_analyst", "")))
                    parsed_pa = artifacts.get("parsed", {}) if isinstance(artifacts.get("parsed"), dict) else {}
                    pa_obj = parsed_pa.get("pattern_analyst")
                    if isinstance(pa_obj, dict):
                        pattern_analyst_trend_champion = str(pa_obj.get("trend_champion") or "")
                        pattern_analyst_seas_champion = str(pa_obj.get("seasonality_champion") or "")
                        pattern_analyst_method_hint = str(pa_obj.get("recommended_method_hint") or "")
                        pattern_analyst_narrative = str(pa_obj.get("cot_narrative") or "")
                except Exception:
                    proposer_think = ""
                    skeptic_think = ""
                    statistician_think = ""
                    pattern_analyst_think = ""
            # If pipeline failed and stored artifacts_path inside description, keep it.
            if not llm_artifacts_path:
                try:
                    desc_obj = json.loads(description) if isinstance(description, str) and description.strip().startswith("{") else None
                    if isinstance(desc_obj, dict) and desc_obj.get("llm_artifacts_path"):
                        llm_artifacts_path = str(desc_obj.get("llm_artifacts_path"))
                except Exception:
                    pass

            # Extract key LLM decision fields into explicit CSV columns.
            desc_obj = None
            try:
                desc_obj = json.loads(description) if isinstance(description, str) and description.strip().startswith("{") else None
            except Exception:
                desc_obj = None

            if isinstance(desc_obj, dict):
                try:
                    score_preset = str(desc_obj.get("score_preset") or "")
                except Exception:
                    score_preset = ""

                tv = desc_obj.get("tool_validation")
                if isinstance(tv, dict):
                    tool_missing = bool(tv.get("tool_missing"))
                    tc = tv.get("tools_called", [])
                    if isinstance(tc, list):
                        try:
                            tools_called_csv = json.dumps(tc, ensure_ascii=False)
                        except Exception:
                            tools_called_csv = str(tc)

                llm = desc_obj.get("llm")
                if isinstance(llm, dict):
                    pr = llm.get("proposer")
                    if isinstance(pr, dict):
                        try:
                            proposer_selected_names = json.dumps(pr.get("selected_names", []), ensure_ascii=False)
                        except Exception:
                            proposer_selected_names = str(pr.get("selected_names", ""))
                        try:
                            proposer_params_overrides = json.dumps(pr.get("params_overrides", {}), ensure_ascii=False)
                        except Exception:
                            proposer_params_overrides = str(pr.get("params_overrides", ""))
                        proposer_force_debate = bool(pr.get("force_debate", False))
                        proposer_debate_margin = pr.get("debate_margin")

                    sk = llm.get("skeptic")
                    if isinstance(sk, dict):
                        try:
                            skeptic_remove_names = json.dumps(sk.get("remove_names", []), ensure_ascii=False)
                        except Exception:
                            skeptic_remove_names = str(sk.get("remove_names", ""))
                        try:
                            skeptic_add_names = json.dumps(sk.get("add_names", []), ensure_ascii=False)
                        except Exception:
                            skeptic_add_names = str(sk.get("add_names", ""))
                        try:
                            skeptic_params_overrides = json.dumps(sk.get("params_overrides", {}), ensure_ascii=False)
                        except Exception:
                            skeptic_params_overrides = str(sk.get("params_overrides", ""))

                    st = llm.get("statistician")
                    if isinstance(st, dict):
                        try:
                            statistician_remove_names = json.dumps(st.get("remove_names", []), ensure_ascii=False)
                        except Exception:
                            statistician_remove_names = str(st.get("remove_names", ""))
                        try:
                            statistician_add_names = json.dumps(st.get("add_names", []), ensure_ascii=False)
                        except Exception:
                            statistician_add_names = str(st.get("add_names", ""))
                        try:
                            statistician_params_overrides = json.dumps(st.get("params_overrides", {}), ensure_ascii=False)
                        except Exception:
                            statistician_params_overrides = str(st.get("params_overrides", ""))

                # Best strategy + final prediction debug trace
                try:
                    b = desc_obj.get("best")
                    if isinstance(b, dict):
                        best_strategy_name = str(b.get("name") or "")
                        params = b.get("params")
                        if isinstance(params, dict):
                            best_strategy_method = str(params.get("method") or "")
                            try:
                                best_strategy_params = json.dumps(params, ensure_ascii=False)
                            except Exception:
                                best_strategy_params = str(params)
                except Exception:
                    pass

                try:
                    pdg = desc_obj.get("predict_debug")
                    if isinstance(pdg, dict):
                        try:
                            predict_debug_csv = json.dumps(pdg, ensure_ascii=False)
                        except Exception:
                            predict_debug_csv = str(pdg)

                        # Collect which base models were selected/used
                        selected = set()
                        if isinstance(pdg.get("chosen_model"), str):
                            selected.add(pdg.get("chosen_model"))
                        if isinstance(pdg.get("chosen_model_by_horizon"), list):
                            for m in pdg.get("chosen_model_by_horizon"):
                                if isinstance(m, str) and m:
                                    selected.add(m)
                        if isinstance(pdg.get("chosen_models_by_horizon"), list):
                            for lst in pdg.get("chosen_models_by_horizon"):
                                if isinstance(lst, list):
                                    for m in lst:
                                        if isinstance(m, str) and m:
                                            selected.add(m)

                        # For weighted methods, also derive selected models from weights
                        wb = pdg.get("weights_by_horizon")
                        if isinstance(wb, dict):
                            for _, wmap in wb.items():
                                if isinstance(wmap, dict):
                                    for m, w in wmap.items():
                                        try:
                                            if float(w) > 0:
                                                selected.add(str(m))
                                        except Exception:
                                            continue
                            try:
                                weights_by_horizon = json.dumps(wb, ensure_ascii=False)
                            except Exception:
                                weights_by_horizon = str(wb)

                        if selected:
                            selected_base_models = json.dumps(sorted(selected), ensure_ascii=False)
                except Exception:
                    pass

            # Final candidates after proposal/debate: from deterministic evaluation ranking.
            try:
                ev = result.get("eval") if isinstance(result, dict) else None
                if isinstance(ev, dict):
                    rk = ev.get("ranking", [])
                    if isinstance(rk, list):
                        names = []
                        for r in rk:
                            if isinstance(r, dict) and r.get("name"):
                                names.append(str(r.get("name")))
                        final_candidate_count = int(len(names))
                        final_candidate_names = json.dumps(names, ensure_ascii=False)
            except Exception:
                pass

            # V2: Oracle comparison fields
            try:
                oracle_info = result.get("oracle") if isinstance(result, dict) else None
                if isinstance(oracle_info, dict):
                    oracle_best_name = str(oracle_info.get("best_name") or "")
                    _obs = oracle_info.get("best_score")
                    oracle_best_score = float(_obs) if _obs is not None else np.nan
                    oracle_best_method = str(oracle_info.get("best_method") or "")
                    _onc = oracle_info.get("n_candidates")
                    oracle_n_candidates = int(_onc) if _onc is not None else np.nan
                    _lit = oracle_info.get("llm_selected_in_oracle_top5")
                    llm_in_oracle_top5 = bool(_lit) if _lit is not None else np.nan
            except Exception:
                pass

            try:
                _delta = result.get("llm_vs_oracle_delta") if isinstance(result, dict) else None
                if _delta is not None:
                    llm_vs_oracle_delta = float(_delta)
            except Exception:
                pass

            # V2: Fixed baselines (equal_weights, best_single)
            try:
                _bl = result.get("baselines") if isinstance(result, dict) else None
                if isinstance(_bl, dict):
                    _ews = _bl.get("equal_weights_score")
                    baseline_equal_weights_score = float(_ews) if _ews is not None else np.nan
                    _bss = _bl.get("best_single_score")
                    baseline_best_single_score = float(_bss) if _bss is not None else np.nan
                    baseline_best_single_model = str(_bl.get("best_single_model") or "")
                    _lew = _bl.get("llm_vs_equal_weights_delta")
                    llm_vs_equal_weights_delta = float(_lew) if _lew is not None else np.nan
                    _lbs = _bl.get("llm_vs_best_single_delta")
                    llm_vs_best_single_delta = float(_lbs) if _lbs is not None else np.nan
            except Exception:
                pass

            # V2: SeriesProfile + StrategyReasoning
            try:
                _sp = result.get("series_profile") if isinstance(result, dict) else None
                if isinstance(_sp, dict) and _sp:
                    series_profile_csv = json.dumps(_sp, ensure_ascii=False)
            except Exception:
                pass

            try:
                _sr = result.get("strategy_reasoning") if isinstance(result, dict) else None
                if isinstance(_sr, dict) and _sr:
                    strategy_reasoning_csv = json.dumps(_sr, ensure_ascii=False)
            except Exception:
                pass

            # V2: think blocks from v2 agents
            try:
                _arts = result.get("llm_artifacts") if isinstance(result, dict) else None
                if isinstance(_arts, dict):
                    raw = _arts.get("raw", {}) or {}
                    series_annotator_think = _extract_think_blocks(str(raw.get("series_annotator", "")))
                    strategy_selector_think = _extract_think_blocks(str(raw.get("strategy_selector", "")))
            except Exception:
                pass

            # V3: pruning + robust combination fields
            try:
                _sv = result.get("survivors") if isinstance(result, dict) else None
                if isinstance(_sv, list):
                    survivors_csv = json.dumps(_sv, ensure_ascii=False)
                regime_v3 = str(result.get("regime") or "")
                _sl = result.get("shrinkage_lambda")
                shrinkage_lambda_v3 = float(_sl) if _sl is not None else np.nan
                _fb = result.get("fellback_to_pruned_mean")
                fellback_to_pruned_mean = bool(_fb) if _fb is not None else np.nan

                _pr = result.get("prune_report") if isinstance(result, dict) else None
                if isinstance(_pr, dict):
                    pruned_models_csv = json.dumps(_pr.get("pruned", []), ensure_ascii=False)
                    prune_blocked_by_mcs_csv = json.dumps(_pr.get("blocked_by_mcs", []), ensure_ascii=False)
                    mcs_superior_set_csv = json.dumps(_pr.get("mcs_superior_set", []), ensure_ascii=False)

                _bl3 = result.get("baselines") if isinstance(result, dict) else None
                if isinstance(_bl3, dict):
                    def _f(key):
                        v = _bl3.get(key)
                        try:
                            return float(v) if v is not None else np.nan
                        except Exception:
                            return np.nan
                    full_mean_score = _f("full_mean_score")
                    full_median_score = _f("full_median_score")
                    pruned_equal_weights_score = _f("pruned_equal_weights_score")
                    pruned_mean_score = _f("pruned_mean_score")
                    pruned_median_score = _f("pruned_median_score")
                    anchor_choice = str(_bl3.get("anchor_choice") or "")
                    llm_regime_score = _f("llm_regime_score")
                    chosen_score_v3 = _f("chosen_score")
                    delta_chosen_vs_full_mean = _f("delta_chosen_vs_full_mean")
                    delta_chosen_vs_full_median = _f("delta_chosen_vs_full_median")
                    delta_chosen_vs_pruned_mean = _f("delta_chosen_vs_pruned_mean")
                    delta_chosen_vs_pruned_median = _f("delta_chosen_vs_pruned_median")
                    delta_pruned_mean_vs_full_mean = _f("delta_pruned_mean_vs_full_mean")
                    try:
                        _pcs = _bl3.get("pool_curated_size")
                        pool_curated_size = int(_pcs) if _pcs is not None else np.nan
                    except Exception:
                        pool_curated_size = np.nan
                    try:
                        _pcr = _bl3.get("pool_curated_removed") or []
                        pool_curated_removed_csv = json.dumps(_pcr, ensure_ascii=False)
                    except Exception:
                        pool_curated_removed_csv = ""
                    oracle_regime_v3 = str(_bl3.get("oracle_regime") or "")
                    _lpb = _bl3.get("llm_picked_best_regime")
                    llm_picked_best_regime = bool(_lpb) if _lpb is not None else np.nan
            except Exception:
                pass

            # V3: think blocks from v3 agents
            try:
                _arts3 = result.get("llm_artifacts") if isinstance(result, dict) else None
                if isinstance(_arts3, dict):
                    raw3 = _arts3.get("raw", {}) or {}
                    series_analyst_think = _extract_think_blocks(str(raw3.get("series_analyst", "")))
                    model_critic_think = _extract_think_blocks(str(raw3.get("model_critic", "")))
                    combination_architect_think = _extract_think_blocks(str(raw3.get("combination_architect", "")))
            except Exception:
                pass

        print("Description: ", description)
        print("Predictions: ", preds_real)

        # In LLM mode, any failure is a hard-stop (no static fallback).
        hard_stop = bool(use_llm and (not result.get("success", False)))

        if preds_real is None:
            preds_real = []

        if hard_stop:
            # Pipeline failed — reset everything
            smape_result = np.nan
            rmse_result = np.nan
            msmape_result = np.nan
            mae_result = np.nan
            mape_result = np.nan
            pocid_result = np.nan
            preds_real = []
            test_arr = np.array([])
        else:
            # Pipeline succeeded — always save preds_real; metrics need test data
            test_arr = np.array(test, dtype=float) if test is not None and len(test) > 0 else np.array([])
            preds_arr = np.array(preds_real, dtype=float) if preds_real else np.array([])

            min_len = min(len(test_arr), len(preds_arr))
            if min_len == 0:
                smape_result = np.nan
                rmse_result = np.nan
                msmape_result = np.nan
                mae_result = np.nan
                mape_result = np.nan
                pocid_result = np.nan
            else:
                test_cut = test_arr[:min_len]
                preds_cut = preds_arr[:min_len]

                smape_result = calculate_smape(preds_cut.reshape(1, -1), test_cut.reshape(1, -1))
                rmse_result = calculate_rmse(preds_cut.reshape(1, -1), test_cut.reshape(1, -1))
                msmape_result = calculate_msmape(preds_cut.reshape(1, -1), test_cut.reshape(1, -1))
                mae_result = calculate_mae(preds_cut.reshape(1, -1), test_cut.reshape(1, -1))
                mape_result = mape(test_cut, preds_cut)
                pocid_result = pocid(test_cut, preds_cut)

        data_serie = {
            "dataset_index": f"{i}",
            "horizon": horizon,
            "regressor": exp_name,
            "mape": mape_result,
            "pocid": pocid_result,
            "smape": smape_result,
            "rmse": rmse_result,
            "msmape": msmape_result,
            "mae": mae_result,
            "test": [test_arr.tolist()],
            "predictions": [list(preds_real) if isinstance(preds_real, (list, np.ndarray)) and len(preds_real) > 0 else []],
            "start_test": "INICIO",
            "final_test": final_test,
            "description": description,
            "debate_ran": debate_ran,
            "debate_trigger": debate_trigger,
            "approach_pre_debate": approach_pre,
            "approach_post_debate": approach_post,
            "debate_explanation": debate_explanation,
            "selection_explanation": selection_explanation,
            "when_good": when_good,
            "decision_report": decision_report,
            "llm_artifacts_path": llm_artifacts_path,
            "score_preset": score_preset,
            "tool_missing": tool_missing,
            "tools_called": tools_called_csv,
            "proposer_selected_names": proposer_selected_names,
            "proposer_params_overrides": proposer_params_overrides,
            "proposer_force_debate": proposer_force_debate,
            "proposer_debate_margin": proposer_debate_margin,
            "final_candidate_names": final_candidate_names,
            "final_candidate_count": final_candidate_count,
            "skeptic_remove_names": skeptic_remove_names,
                "skeptic_add_names": skeptic_add_names,
            "skeptic_params_overrides": skeptic_params_overrides,
            "statistician_remove_names": statistician_remove_names,
                "statistician_add_names": statistician_add_names,
            "statistician_params_overrides": statistician_params_overrides,

            "best_strategy_name": best_strategy_name,
            "best_strategy_method": best_strategy_method,
            "best_strategy_params": best_strategy_params,
            "predict_debug": predict_debug_csv,
            "selected_base_models": selected_base_models,
            "weights_by_horizon": weights_by_horizon,
            "proposer_think": proposer_think,
            "skeptic_think": skeptic_think,
            "statistician_think": statistician_think,
            "pattern_analyst_think": pattern_analyst_think,
            "pattern_analyst_trend_champion": pattern_analyst_trend_champion,
            "pattern_analyst_seas_champion": pattern_analyst_seas_champion,
            "pattern_analyst_method_hint": pattern_analyst_method_hint,
            "pattern_analyst_narrative": pattern_analyst_narrative,

            # V2: Oracle comparison
            "oracle_best_name": oracle_best_name,
            "oracle_best_score": oracle_best_score,
            "oracle_best_method": oracle_best_method,
            "oracle_n_candidates": oracle_n_candidates,
            "llm_vs_oracle_delta": llm_vs_oracle_delta,
            "llm_in_oracle_top5": llm_in_oracle_top5,

            # V2: Fixed baselines (publication evidence — must beat equal_weights)
            "baseline_equal_weights_score": baseline_equal_weights_score,
            "baseline_best_single_score": baseline_best_single_score,
            "baseline_best_single_model": baseline_best_single_model,
            "llm_vs_equal_weights_delta": llm_vs_equal_weights_delta,
            "llm_vs_best_single_delta": llm_vs_best_single_delta,

            # V2: Structured annotations
            "series_profile": series_profile_csv,
            "strategy_reasoning": strategy_reasoning_csv,
            "series_annotator_think": series_annotator_think,
            "strategy_selector_think": strategy_selector_think,

            # V3: Pruning + robust combination
            "survivors": survivors_csv,
            "pruned_models": pruned_models_csv,
            "prune_blocked_by_mcs": prune_blocked_by_mcs_csv,
            "mcs_superior_set": mcs_superior_set_csv,
            "regime": regime_v3,
            "shrinkage_lambda": shrinkage_lambda_v3,
            "fellback_to_pruned_mean": fellback_to_pruned_mean,
            "oracle_regime": oracle_regime_v3,
            "llm_picked_best_regime": llm_picked_best_regime,
            "full_mean_score": full_mean_score,
            "full_median_score": full_median_score,
            "pruned_equal_weights_score": pruned_equal_weights_score,
            "llm_regime_score": llm_regime_score,
            "chosen_score": chosen_score_v3,
            "delta_chosen_vs_full_mean": delta_chosen_vs_full_mean,
            "delta_chosen_vs_full_median": delta_chosen_vs_full_median,
            "delta_chosen_vs_pruned_mean": delta_chosen_vs_pruned_mean,
            "delta_pruned_mean_vs_full_mean": delta_pruned_mean_vs_full_mean,
            # Sprint-1
            "pruned_mean_score": pruned_mean_score,
            "pruned_median_score": pruned_median_score,
            "anchor_choice": anchor_choice,
            "delta_chosen_vs_pruned_median": delta_chosen_vs_pruned_median,
            "pool_curated_size": pool_curated_size,
            "pool_curated_removed": pool_curated_removed_csv,
            "series_analyst_think": series_analyst_think,
            "model_critic_think": model_critic_think,
            "combination_architect_think": combination_architect_think,
        }

        df_new = pd.DataFrame(data_serie)
        df_new = df_new.reindex(columns=cols_serie)
        df_new.to_csv(path_csv, sep=";", mode="a", header=False, index=False)

        if hard_stop:
            raise RuntimeError(f"LLM run failed at dataset_index={i}. See CSV row description for details.")

import orchestrator.utils as _utils
if __name__ == "__main__":
    models = [
        "ARIMA",
        "ETS",
        "THETA",
        #"ridge",
        "rf",
        "catboost",
        #"CWT_ridge",
        #"DWT_ridge",
        #"FT_ridge",
        "CWT_rf",
        "DWT_rf",
        "FT_rf",
        "CWT_catboost",
        "DWT_catboost",
        "FT_catboost",
        "ONLY_CWT_catboost",
        "ONLY_CWT_rf",
        #"ONLY_CWT_ridge",
        "ONLY_DWT_catboost",
        "ONLY_DWT_rf",
        #"ONLY_DWT_ridge",
        "ONLY_FT_catboost",
        "ONLY_FT_rf",
        #"ONLY_FT_ridge",
        "NaiveSeasonal",
        "NaiveMovingAverage",
    ]

    dataset = "NN5_WEEKLY_DATASET"
    original_tsf_path = "../forecasting_datasets/nn5_weekly_dataset.tsf"
    
    exec_dataset_orchestrator(
        models,
        dataset=dataset,
        use_llm=True,
        series_analyst_model=_utils.ModelConfig(model="qwen3:14b", temperature=0.0),
        model_critic_model=_utils.ModelConfig(model="qwen3:14b", temperature=0.0),
        combination_architect_model=_utils.ModelConfig(model="qwen3:14b", temperature=0.0),
        debug=False,
        rolling="expanding",
        train_window=3,
        llm_logs=True,
        version="v3_pruning",
        original_tsf_path=original_tsf_path,
    )
