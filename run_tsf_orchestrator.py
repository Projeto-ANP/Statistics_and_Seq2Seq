import os
import re
import json
import pandas as pd
import numpy as np
from streamfuels.datasets import DatasetLoader
from sklearn.metrics import mean_absolute_percentage_error as mape
from orchestrator_langchain.context import read_model_preds
from all_functions import (
    calculate_smape,
    calculate_rmse,
    calculate_msmape,
    calculate_mae,
    pocid,
)


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


# ---------------------------------------------------------------------------
# CSV schema — split by pipeline mode.
#   COLS_CORE      : sempre presentes (identificação, métricas, descrição).
#   COLS_LEGACY    : apenas quando `use_halmoc=False` (Pattern Analyst /
#                    Proposer / Skeptic / Statistician + debate).
#   COLS_HALMOC    : apenas quando `use_halmoc=True` (Diagnostician /
#                    Council / Judge + Meta-Combiner + MCS + Conformal).
# O `cols_used` é definido em runtime dentro de `exec_dataset_orchestrator`.
# ---------------------------------------------------------------------------

COLS_CORE = [
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
]

COLS_LEGACY = [
    "debate_ran",
    "debate_trigger",
    "approach_pre_debate",
    "approach_post_debate",
    "debate_explanation",
    "selection_explanation",
    "when_good",
    "decision_report",
    "llm_artifacts_path",
    "score_preset",
    "tool_missing",
    "tools_called",
    "proposer_selected_names",
    "proposer_params_overrides",
    "proposer_force_debate",
    "proposer_debate_margin",
    "final_candidate_names",
    "final_candidate_count",
    "skeptic_remove_names",
    "skeptic_add_names",
    "skeptic_params_overrides",
    "statistician_remove_names",
    "statistician_add_names",
    "statistician_params_overrides",
    "best_strategy_name",
    "best_strategy_method",
    "best_strategy_params",
    "predict_debug",
    "selected_base_models",
    "weights_by_horizon",
    "proposer_think",
    "skeptic_think",
    "statistician_think",
    "pattern_analyst_think",
    "pattern_analyst_trend_champion",
    "pattern_analyst_seas_champion",
    "pattern_analyst_method_hint",
    "pattern_analyst_narrative",
]

COLS_HALMOC = [
    "llm_artifacts_path",
    "halmoc_dataset_id",
    "halmoc_diagnosis",
    "halmoc_council_proposals",
    "halmoc_judge_confidence",
    "halmoc_judge_notes",
    "halmoc_mcs_surviving",
    "halmoc_mcs_eliminated",
    "halmoc_winning_combiner",
    "halmoc_winning_params",
    "halmoc_weights_by_horizon",
    "halmoc_selected_base_models",
    "halmoc_conformal_method",
    "halmoc_conformal_coverage",
    "halmoc_conformal_width",
    "halmoc_memory_path",
    "halmoc_final_composite",
    # Per-agent auditability (which model answered, and a preview of its
    # <think> reasoning / output).
    "halmoc_diagnostician_model",
    "halmoc_diagnostician_think",
    "halmoc_council_summary",
    "halmoc_judge_model",
    "halmoc_judge_think",
    "halmoc_judge_ranking",
]

# Backwards-compatibility: some callers still import `COLS_SERIE`.  Keep it as
# the *union* of the three groups so re-indexing on any existing file will not
# drop columns.  The active schema per run is `cols_used`, computed below.
COLS_SERIE = (
    COLS_CORE
    + [c for c in COLS_LEGACY if c not in COLS_CORE]
    + [c for c in COLS_HALMOC if c not in COLS_CORE and c not in COLS_LEGACY]
)


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

    final_test_date = pd.to_datetime(
        final_test, errors="coerce", infer_datetime_format=True
    )

    for model in models:
        df = read_model_preds(model, dataset_index, dataset=dataset)
        test_df = df[df["final_test"] == final_test_date]

        if not test_df.empty:
            final_row = test_df.iloc[0]
            final_test_predictions[model] = extract_values(final_row["predictions"])
            final_test_data = extract_values(final_row["test"])

    return final_test_predictions, final_test_data


def exec_dataset_orchestrator(
    models,
    dataset,
    use_llm: bool = False,
    ollama_model: str = "mychen76/qwen3_cline_roocode:14b",
    debug: bool = False,
    rolling: str = "expanding",
    train_window: int = 3,
    llm_logs: bool = True,
    # start_index: int = 0,
    # end_index: int = 182,
    version: str = "v1_halmoc",
    use_halmoc: bool = True,
    halmoc_council_model_ids=None,
    halmoc_diagnostician_model_id: str | None = None,
    halmoc_judge_model_id: str | None = None,
    halmoc_conformal_name: str = "aci",
    halmoc_conformal_alpha: float = 0.10,
    halmoc_mcs_alpha: float = 0.10,
    halmoc_max_proposals: int = 6,
):
    # Per-agent model ids fall back to the shared `ollama_model` when not
    # set — so the legacy single-model call still works unchanged.
    halmoc_diagnostician_model_id = halmoc_diagnostician_model_id or ollama_model
    halmoc_judge_model_id = halmoc_judge_model_id or ollama_model
    # dataset = "ANP_MONTHLY"
    # dataset = "ETTH1"
    dataset_file = f"./timeseries/mestrado/resultados/catboost/normal/{dataset}.csv"
    df_dt = pd.read_csv(dataset_file, sep=";")

    df_dt["final_test"] = pd.to_datetime(
        df_dt["final_test"],
        errors="coerce",  # transforma inválidos em NaT
        infer_datetime_format=True,
    )

    if df_dt["final_test"].isna().sum() > 0:
        print("Existem datas inválidas que viraram NaT.")

    df_new_dt = df_dt.sort_values("final_test").reset_index(drop=True)
    # df_new_dt.iloc[-1]['horizon']
    if use_halmoc:
        exp_suffix = "halmoc" if use_llm else "halmoc_noLLM"
    else:
        exp_suffix = "llm" if use_llm else "deterministic"
    exp_name = f"orchestrator_{exp_suffix}_{version}"
    horizon = df_new_dt.iloc[-1]["horizon"]
    final_test = df_new_dt.iloc[-1]["final_test"]
    num_series = df_dt["dataset_index"].nunique()

    # Select the active CSV schema for this run (mode-specific).
    cols_used = list(COLS_CORE) + list(COLS_HALMOC if use_halmoc else COLS_LEGACY)

    path_experiments = f"./timeseries/mestrado/resultados/{exp_name}/"
    path_csv = f"{path_experiments}/{dataset}.csv"
    path_llm_artifacts = f"{path_experiments}/llm_artifacts/{dataset}/"
    os.makedirs(path_experiments, exist_ok=True)
    os.makedirs(path_llm_artifacts, exist_ok=True)

    from orchestrator_langchain.context import (
        CONTEXT_MEMORY,
        generate_all_validations_context,
        init_context,
    )
    from orchestrator.pipeline import run_deterministic_pipeline, run_llm_pipeline
    from orchestrator_langchain.pipeline import run_langchain_pipeline
    from orchestrator_langchain.halmoc_pipeline import run_halmoc_langchain_pipeline

    # Ensure CSV schema is up-to-date (add missing columns if file already exists).
    if not os.path.exists(path_csv):
        pd.DataFrame(columns=cols_used).to_csv(path_csv, sep=";", index=False)
    else:
        try:
            df_existing = pd.read_csv(path_csv, sep=";")
            missing = [c for c in cols_used if c not in df_existing.columns]
            if missing:
                for c in missing:
                    df_existing[c] = np.nan
                df_existing = df_existing.reindex(columns=cols_used)
                df_existing.to_csv(path_csv, sep=";", index=False)
        except Exception:
            # If the existing file is malformed, keep running; new rows will still append.
            pass

    for i in range(num_series):
        init_context()
        CONTEXT_MEMORY["models_available"] = models
        generate_all_validations_context(
            models, i, train_window=train_window, dataset=dataset
        )
        print(f"----- DATASET INDEX: {i} -----")
        if use_llm:
            try:
                if use_halmoc:
                    result = run_halmoc_langchain_pipeline(
                        model_id=ollama_model,
                        debug=debug,
                        rolling_mode=rolling,
                        train_window=train_window,
                        require_tool_call=True,
                        llm_logs=llm_logs,
                        use_llm=True,
                        diagnostician_model_id=halmoc_diagnostician_model_id,
                        council_model_ids=halmoc_council_model_ids,
                        judge_model_id=halmoc_judge_model_id,
                        mcs_alpha=halmoc_mcs_alpha,
                        conformal_name=halmoc_conformal_name,
                        conformal_alpha=halmoc_conformal_alpha,
                        max_proposals=halmoc_max_proposals,
                    )
                else:
                    result = run_langchain_pipeline(
                        model_id=ollama_model,
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
                    llm_artifacts_path = os.path.abspath(
                        os.path.join(path_llm_artifacts, f"dataset_{i}.json")
                    )
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
            if use_halmoc:
                result = run_halmoc_langchain_pipeline(
                    model_id=ollama_model,
                    debug=debug,
                    rolling_mode=rolling,
                    train_window=train_window,
                    require_tool_call=False,
                    llm_logs=llm_logs,
                    use_llm=False,
                    mcs_alpha=halmoc_mcs_alpha,
                    conformal_name=halmoc_conformal_name,
                    conformal_alpha=halmoc_conformal_alpha,
                    max_proposals=halmoc_max_proposals,
                )
            else:
                result = run_deterministic_pipeline()

        _, test = get_predictions_models(
            models, dataset_index=i, final_test=final_test, dataset=dataset.upper()
        )

        description = result.get("description", "")
        if not isinstance(description, str):
            try:
                description = json.dumps(description, ensure_ascii=False)
            except Exception:
                description = str(description)
        preds_real = result.get("result", [])

        # Shared across both modes.
        llm_artifacts_path = ""

        # Legacy columns (only populated when use_halmoc=False).
        debate_explanation = ""
        selection_explanation = ""
        when_good = ""
        decision_report = ""
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
        debate_ran = np.nan
        debate_trigger = np.nan
        approach_pre = np.nan
        approach_post = np.nan

        if use_llm and not use_halmoc:
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
                debate_explanation = str(
                    expl.get("orchestrator_debate_notes")
                    or expl.get("skeptic_rationale")
                    or ""
                )
                selection_explanation = str(expl.get("orchestrator_reasoning") or "")
                when_good = str(
                    expl.get("orchestrator_when_good")
                    or expl.get("statistician_when_good")
                    or expl.get("skeptic_when_good")
                    or ""
                )

            try:
                decision_report = (
                    f"pre={approach_pre} | post={approach_post} | score_preset={score_preset} | debate_ran={debate_ran} | trigger={debate_trigger} | "
                    f"debate_note={debate_explanation} | selection={selection_explanation} | when_good={when_good}"
                )
            except Exception:
                decision_report = ""

            # Persist full LLM artifacts for auditability (raw prompts/outputs).
            artifacts = (
                result.get("llm_artifacts") if isinstance(result, dict) else None
            )
            if isinstance(artifacts, dict):
                try:
                    llm_artifacts_path = os.path.abspath(
                        os.path.join(path_llm_artifacts, f"dataset_{i}.json")
                    )
                    with open(llm_artifacts_path, "w", encoding="utf-8") as f:
                        json.dump(
                            {"dataset_index": i, "artifacts": artifacts},
                            f,
                            ensure_ascii=False,
                            indent=2,
                        )
                except Exception:
                    llm_artifacts_path = ""
                try:
                    raw = (
                        artifacts.get("raw", {})
                        if isinstance(artifacts.get("raw"), dict)
                        else {}
                    )
                    proposer_think = _extract_think_blocks(str(raw.get("proposer", "")))
                    skeptic_think = _extract_think_blocks(str(raw.get("skeptic", "")))
                    statistician_think = _extract_think_blocks(
                        str(raw.get("statistician", ""))
                    )
                    pattern_analyst_think = _extract_think_blocks(
                        str(raw.get("pattern_analyst", ""))
                    )
                    parsed_pa = (
                        artifacts.get("parsed", {})
                        if isinstance(artifacts.get("parsed"), dict)
                        else {}
                    )
                    pa_obj = parsed_pa.get("pattern_analyst")
                    if isinstance(pa_obj, dict):
                        pattern_analyst_trend_champion = str(
                            pa_obj.get("trend_champion") or ""
                        )
                        pattern_analyst_seas_champion = str(
                            pa_obj.get("seasonality_champion") or ""
                        )
                        pattern_analyst_method_hint = str(
                            pa_obj.get("recommended_method_hint") or ""
                        )
                        pattern_analyst_narrative = str(
                            pa_obj.get("cot_narrative") or ""
                        )
                except Exception:
                    proposer_think = ""
                    skeptic_think = ""
                    statistician_think = ""
                    pattern_analyst_think = ""
            # If pipeline failed and stored artifacts_path inside description, keep it.
            if not llm_artifacts_path:
                try:
                    desc_obj = (
                        json.loads(description)
                        if isinstance(description, str)
                        and description.strip().startswith("{")
                        else None
                    )
                    if isinstance(desc_obj, dict) and desc_obj.get(
                        "llm_artifacts_path"
                    ):
                        llm_artifacts_path = str(desc_obj.get("llm_artifacts_path"))
                except Exception:
                    pass

            # Extract key LLM decision fields into explicit CSV columns.
            desc_obj = None
            try:
                desc_obj = (
                    json.loads(description)
                    if isinstance(description, str)
                    and description.strip().startswith("{")
                    else None
                )
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
                            proposer_selected_names = json.dumps(
                                pr.get("selected_names", []), ensure_ascii=False
                            )
                        except Exception:
                            proposer_selected_names = str(pr.get("selected_names", ""))
                        try:
                            proposer_params_overrides = json.dumps(
                                pr.get("params_overrides", {}), ensure_ascii=False
                            )
                        except Exception:
                            proposer_params_overrides = str(
                                pr.get("params_overrides", "")
                            )
                        proposer_force_debate = bool(pr.get("force_debate", False))
                        proposer_debate_margin = pr.get("debate_margin")

                    sk = llm.get("skeptic")
                    if isinstance(sk, dict):
                        try:
                            skeptic_remove_names = json.dumps(
                                sk.get("remove_names", []), ensure_ascii=False
                            )
                        except Exception:
                            skeptic_remove_names = str(sk.get("remove_names", ""))
                        try:
                            skeptic_add_names = json.dumps(
                                sk.get("add_names", []), ensure_ascii=False
                            )
                        except Exception:
                            skeptic_add_names = str(sk.get("add_names", ""))
                        try:
                            skeptic_params_overrides = json.dumps(
                                sk.get("params_overrides", {}), ensure_ascii=False
                            )
                        except Exception:
                            skeptic_params_overrides = str(
                                sk.get("params_overrides", "")
                            )

                    st = llm.get("statistician")
                    if isinstance(st, dict):
                        try:
                            statistician_remove_names = json.dumps(
                                st.get("remove_names", []), ensure_ascii=False
                            )
                        except Exception:
                            statistician_remove_names = str(st.get("remove_names", ""))
                        try:
                            statistician_add_names = json.dumps(
                                st.get("add_names", []), ensure_ascii=False
                            )
                        except Exception:
                            statistician_add_names = str(st.get("add_names", ""))
                        try:
                            statistician_params_overrides = json.dumps(
                                st.get("params_overrides", {}), ensure_ascii=False
                            )
                        except Exception:
                            statistician_params_overrides = str(
                                st.get("params_overrides", "")
                            )

                # Best strategy + final prediction debug trace
                try:
                    b = desc_obj.get("best")
                    if isinstance(b, dict):
                        best_strategy_name = str(b.get("name") or "")
                        params = b.get("params")
                        if isinstance(params, dict):
                            best_strategy_method = str(params.get("method") or "")
                            try:
                                best_strategy_params = json.dumps(
                                    params, ensure_ascii=False
                                )
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
                            selected_base_models = json.dumps(
                                sorted(selected), ensure_ascii=False
                            )
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
            test_arr = (
                np.array(test, dtype=float)
                if test is not None and len(test) > 0
                else np.array([])
            )
            preds_arr = (
                np.array(preds_real, dtype=float) if preds_real else np.array([])
            )

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

                smape_result = calculate_smape(
                    preds_cut.reshape(1, -1), test_cut.reshape(1, -1)
                )
                rmse_result = calculate_rmse(
                    preds_cut.reshape(1, -1), test_cut.reshape(1, -1)
                )
                msmape_result = calculate_msmape(
                    preds_cut.reshape(1, -1), test_cut.reshape(1, -1)
                )
                mae_result = calculate_mae(
                    preds_cut.reshape(1, -1), test_cut.reshape(1, -1)
                )
                mape_result = mape(test_cut, preds_cut)
                pocid_result = pocid(test_cut, preds_cut)

        # --- HALMOC-specific CSV columns (only populated when use_halmoc=True) ---
        halmoc_dataset_id = ""
        halmoc_diagnosis_csv = ""
        halmoc_council_proposals_csv = ""
        halmoc_judge_confidence = ""
        halmoc_judge_notes = ""
        halmoc_mcs_surviving_csv = ""
        halmoc_mcs_eliminated_csv = ""
        halmoc_winning_combiner = ""
        halmoc_winning_params = ""
        halmoc_weights_by_horizon_csv = ""
        halmoc_selected_base_models_csv = ""
        halmoc_conformal_method_csv = ""
        halmoc_conformal_coverage_csv = np.nan
        halmoc_conformal_width_csv = np.nan
        halmoc_memory_path_csv = ""
        halmoc_final_composite_csv = np.nan
        halmoc_diagnostician_model_csv = ""
        halmoc_diagnostician_think_csv = ""
        halmoc_council_summary_csv = ""
        halmoc_judge_model_csv = ""
        halmoc_judge_think_csv = ""
        halmoc_judge_ranking_csv = ""
        if use_halmoc:
            try:
                halmoc_raw = result.get("halmoc") if isinstance(result, dict) else None
                if isinstance(halmoc_raw, dict) and halmoc_raw:
                    halmoc_dataset_id = str(halmoc_raw.get("dataset_id") or "")
                    diag = halmoc_raw.get("diagnosis") or {}
                    halmoc_diagnosis_csv = (
                        json.dumps(diag, ensure_ascii=False) if diag else ""
                    )
                    cps = halmoc_raw.get("council_proposals") or []
                    halmoc_council_proposals_csv = (
                        json.dumps(cps, ensure_ascii=False) if cps else ""
                    )
                    jd = halmoc_raw.get("judge") or {}
                    halmoc_judge_confidence = str(jd.get("confidence") or "")
                    halmoc_judge_notes = str(jd.get("tie_resolution_notes") or "")
                    mcs_obj = halmoc_raw.get("mcs") or {}
                    halmoc_mcs_surviving_csv = json.dumps(
                        mcs_obj.get("surviving", []), ensure_ascii=False
                    )
                    halmoc_mcs_eliminated_csv = json.dumps(
                        mcs_obj.get("eliminated_order", []), ensure_ascii=False
                    )
                    wp = halmoc_raw.get("winner_proposal") or {}
                    halmoc_winning_combiner = str(wp.get("combiner_name") or "")
                    halmoc_winning_params = json.dumps(
                        wp.get("params", {}), ensure_ascii=False
                    )
                    conf = halmoc_raw.get("conformal") or {}
                    halmoc_conformal_method_csv = str(conf.get("method") or "")
                    if "empirical_coverage" in conf:
                        try:
                            halmoc_conformal_coverage_csv = float(
                                conf.get("empirical_coverage")
                            )
                        except Exception:
                            halmoc_conformal_coverage_csv = np.nan
                    if "mean_width" in conf:
                        try:
                            halmoc_conformal_width_csv = float(conf.get("mean_width"))
                        except Exception:
                            halmoc_conformal_width_csv = np.nan
                    halmoc_memory_path_csv = str(halmoc_raw.get("memory_path") or "")
                    fs = halmoc_raw.get("final_score") or {}
                    try:
                        halmoc_final_composite_csv = float(fs.get("composite", np.nan))
                    except Exception:
                        halmoc_final_composite_csv = np.nan

                    # Per-horizon weights (from linear combiners).  Map each
                    # horizon to {model_name: weight}.
                    wph = halmoc_raw.get("winning_weights_per_horizon")
                    model_names = halmoc_raw.get("model_names") or []
                    if isinstance(wph, list) and model_names:
                        weights_map = {}
                        for h_idx, row in enumerate(wph):
                            if isinstance(row, list):
                                weights_map[f"h{h_idx+1}"] = {
                                    str(m): float(w)
                                    for m, w in zip(model_names, row)
                                    if isinstance(w, (int, float))
                                }
                        if weights_map:
                            halmoc_weights_by_horizon_csv = json.dumps(
                                weights_map, ensure_ascii=False
                            )

                    mns = halmoc_raw.get("selected_base_models") or model_names
                    if mns:
                        halmoc_selected_base_models_csv = json.dumps(
                            list(mns), ensure_ascii=False
                        )

                    # Per-agent model ids + <think> previews for CSV audit.
                    halmoc_diagnostician_model_csv = str(
                        halmoc_raw.get("diagnostician_model_id") or ""
                    )
                    halmoc_diagnostician_think_csv = str(
                        halmoc_raw.get("diagnostician_think") or ""
                    )
                    halmoc_judge_model_csv = str(
                        halmoc_raw.get("judge_model_id") or ""
                    )
                    halmoc_judge_think_csv = str(
                        halmoc_raw.get("judge_think") or ""
                    )
                    try:
                        ranked = jd.get("ranked_proposals", [])
                        if isinstance(ranked, list):
                            halmoc_judge_ranking_csv = json.dumps(
                                ranked, ensure_ascii=False
                            )
                    except Exception:
                        halmoc_judge_ranking_csv = ""

                    council_raws = halmoc_raw.get("council_raws") or []
                    if council_raws:
                        summary = []
                        for cr in council_raws:
                            think_preview = str(cr.get("think") or "")
                            if len(think_preview) > 400:
                                think_preview = think_preview[:400] + "…"
                            props = cr.get("proposals") or []
                            prop_names = [
                                str(p.get("combiner_name", "?")) for p in props
                            ]
                            summary.append({
                                "member_idx": cr.get("member_idx"),
                                "model_id": cr.get("model_id"),
                                "duration_s": cr.get("duration_s"),
                                "think_preview": think_preview,
                                "proposal_names": prop_names,
                            })
                        try:
                            halmoc_council_summary_csv = json.dumps(
                                summary, ensure_ascii=False
                            )
                        except Exception:
                            halmoc_council_summary_csv = str(summary)

                # Persist full HALMOC artefacts for auditability (diagnosis,
                # proposals, judge decision, MCS + conformal) — same folder
                # the legacy pipeline used, different JSON shape.
                artifacts = (
                    result.get("llm_artifacts") if isinstance(result, dict) else None
                )
                if isinstance(artifacts, dict):
                    try:
                        llm_artifacts_path = os.path.abspath(
                            os.path.join(path_llm_artifacts, f"dataset_{i}.json")
                        )
                        with open(llm_artifacts_path, "w", encoding="utf-8") as f:
                            json.dump(
                                {
                                    "dataset_index": i,
                                    "mode": "halmoc",
                                    "artifacts": artifacts,
                                },
                                f,
                                ensure_ascii=False,
                                indent=2,
                            )
                    except Exception:
                        llm_artifacts_path = ""
            except Exception:
                # Never let CSV-field extraction crash the loop.
                pass

        # Build row dict using only the columns for the active mode.
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
            "predictions": [
                (
                    list(preds_real)
                    if isinstance(preds_real, (list, np.ndarray))
                    and len(preds_real) > 0
                    else []
                )
            ],
            "start_test": "INICIO",
            "final_test": final_test,
            "description": description,
        }

        if use_halmoc:
            data_serie.update(
                {
                    "llm_artifacts_path": llm_artifacts_path,
                    "halmoc_dataset_id": halmoc_dataset_id,
                    "halmoc_diagnosis": halmoc_diagnosis_csv,
                    "halmoc_council_proposals": halmoc_council_proposals_csv,
                    "halmoc_judge_confidence": halmoc_judge_confidence,
                    "halmoc_judge_notes": halmoc_judge_notes,
                    "halmoc_mcs_surviving": halmoc_mcs_surviving_csv,
                    "halmoc_mcs_eliminated": halmoc_mcs_eliminated_csv,
                    "halmoc_winning_combiner": halmoc_winning_combiner,
                    "halmoc_winning_params": halmoc_winning_params,
                    "halmoc_weights_by_horizon": halmoc_weights_by_horizon_csv,
                    "halmoc_selected_base_models": halmoc_selected_base_models_csv,
                    "halmoc_conformal_method": halmoc_conformal_method_csv,
                    "halmoc_conformal_coverage": halmoc_conformal_coverage_csv,
                    "halmoc_conformal_width": halmoc_conformal_width_csv,
                    "halmoc_memory_path": halmoc_memory_path_csv,
                    "halmoc_final_composite": halmoc_final_composite_csv,
                    "halmoc_diagnostician_model": halmoc_diagnostician_model_csv,
                    "halmoc_diagnostician_think": halmoc_diagnostician_think_csv,
                    "halmoc_council_summary": halmoc_council_summary_csv,
                    "halmoc_judge_model": halmoc_judge_model_csv,
                    "halmoc_judge_think": halmoc_judge_think_csv,
                    "halmoc_judge_ranking": halmoc_judge_ranking_csv,
                }
            )
        else:
            data_serie.update(
                {
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
                }
            )

        df_new = pd.DataFrame(data_serie)
        df_new = df_new.reindex(columns=cols_used)
        df_new.to_csv(path_csv, sep=";", mode="a", header=False, index=False)

        if hard_stop:
            raise RuntimeError(
                f"LLM run failed at dataset_index={i}. See CSV row description for details."
            )


if __name__ == "__main__":
    models = [
        "ARIMA",
        "ETS",
        "THETA",
        "ridge",
        "rf",
        "catboost",
        "CWT_ridge",
        "DWT_ridge",
        "FT_ridge",
        "CWT_rf",
        "DWT_rf",
        "FT_rf",
        "CWT_catboost",
        "DWT_catboost",
        "FT_catboost",
        "ONLY_CWT_catboost",
        "ONLY_CWT_rf",
        "ONLY_CWT_ridge",
        "ONLY_DWT_catboost",
        "ONLY_DWT_rf",
        "ONLY_DWT_ridge",
        "ONLY_FT_catboost",
        "ONLY_FT_rf",
        "ONLY_FT_ridge",
        "NaiveSeasonal",
        "NaiveMovingAverage",
    ]

    dataset = "US_BIRTHS_DATASET"
    # Example: heterogeneous council (one qwen3, one gpt-oss, one llama).
    # Set to `None` to reuse `ollama_model` for every agent.
    exec_dataset_orchestrator(
        models,
        dataset=dataset,
        use_llm=True,
        ollama_model="mychen76/qwen3_cline_roocode:14b",  # default/fallback
        halmoc_diagnostician_model_id="mychen76/qwen3_cline_roocode:14b",
        halmoc_council_model_ids=[
            # "mychen76/qwen3_cline_roocode:14b", # ollama pull gemma4:26b
            # "mychen76/qwen3_cline_roocode:14b", # ollama pull glm-4.7-flash:q4_K_M
            # "mychen76/qwen3_cline_roocode:14b", # ollama pull gpt-oss:20b
            "gpt-oss:20b",
            "glm-4.7-flash:q4_K_M",
            "gemma4:26b"
        ],
        halmoc_judge_model_id="mychen76/qwen3_cline_roocode:14b",
        debug=False,
        rolling="expanding",
        train_window=5,  # n_windows = train_window - 1 (blind row); HALMOC exige ≥ 3
        llm_logs=True,
    )
