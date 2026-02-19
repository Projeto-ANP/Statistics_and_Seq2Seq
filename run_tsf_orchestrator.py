import os
import re
import json
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error as mape

from all_functions import calculate_smape, calculate_rmse, calculate_msmape, calculate_mae, pocid


def extract_values(list_str):
    if isinstance(list_str, str):
        numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", list_str)
        return [float(num) for num in numbers]
    return []


def read_model_preds(model_name, dataset_index):
    df = pd.read_csv(
        f"./timeseries/mestrado/resultados/{model_name}/normal/ANP_MONTHLY.csv",
        sep=";",
    )
    df = df[df["dataset_index"] == dataset_index]

    df["start_test"] = pd.to_datetime(df["start_test"], format="%Y-%m-%d")
    df["final_test"] = pd.to_datetime(df["final_test"], format="%Y-%m-%d")
    df = df.sort_values(by="start_test")

    return df


COLS_SERIE = [
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

    # Traceability of the final combination applied on final_test
    "best_strategy_name",
    "best_strategy_method",
    "best_strategy_params",
    "predict_debug",
    "selected_base_models",
    "weights_by_horizon",

    # LLM raw think blocks (if present)
    "proposer_think",
    "skeptic_think",
    "statistician_think",
]


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


def get_predictions_models(models, dataset_index, final_test):
    final_test_predictions = {}
    final_test_data = None

    final_test_date = pd.to_datetime(final_test, format="%Y-%m-%d")

    for model in models:
        df = read_model_preds(model, dataset_index)
        test_df = df[df["final_test"] == final_test_date]

        if not test_df.empty:
            final_row = test_df.iloc[0]
            final_test_predictions[model] = extract_values(final_row["predictions"])
            final_test_data = extract_values(final_row["test"])

    return final_test_predictions, final_test_data


def exec_dataset_orchestrator(
    models,
    use_llm: bool = False,
    ollama_model: str = "mychen76/qwen3_cline_roocode:14b",
    debug: bool = False,
    rolling: str = "expanding",
    train_window: int = 5,
    llm_logs: bool = True,
    start_index: int = 0,
    end_index: int = 182,
    version: str = "v3_ade",
):
    dataset = "ANP_MONTHLY"
    exp_name = f"orchestrator_llm_{version}" if use_llm else f"orchestrator_deterministic_{version}"
    horizon = 12
    final_test = "2024-11-30"

    path_experiments = f"./timeseries/mestrado/resultados/{exp_name}/"
    path_csv = f"{path_experiments}/{dataset}.csv"
    path_llm_artifacts = f"{path_experiments}/llm_artifacts/{dataset}/"
    os.makedirs(path_experiments, exist_ok=True)
    os.makedirs(path_llm_artifacts, exist_ok=True)

    from agent.context import CONTEXT_MEMORY, generate_all_validations_context, init_context
    from orchestrator.pipeline import run_deterministic_pipeline, run_llm_pipeline
    from orchestrator_langchain.pipeline import run_langchain_pipeline

    # Ensure CSV schema is up-to-date (add missing columns if file already exists).
    if not os.path.exists(path_csv):
        pd.DataFrame(columns=COLS_SERIE).to_csv(path_csv, sep=";", index=False)
    else:
        try:
            df_existing = pd.read_csv(path_csv, sep=";")
            missing = [c for c in COLS_SERIE if c not in df_existing.columns]
            if missing:
                for c in missing:
                    df_existing[c] = np.nan
                df_existing = df_existing.reindex(columns=COLS_SERIE)
                df_existing.to_csv(path_csv, sep=";", index=False)
        except Exception:
            # If the existing file is malformed, keep running; new rows will still append.
            pass

    for i in range(int(start_index), int(end_index)):
        init_context()
        CONTEXT_MEMORY["models_available"] = models
        generate_all_validations_context(models, i)
        print(f"----- DATASET INDEX: {i} -----")
        if use_llm:
            try:
                # result = run_llm_pipeline(
                #     model_id=ollama_model,
                #     debug=debug,
                #     rolling_mode=rolling,
                #     train_window=train_window,
                #     require_tool_call=True,
                #     llm_logs=llm_logs,
                # )
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

        _, test = get_predictions_models(models, dataset_index=i, final_test=final_test)

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
                except Exception:
                    proposer_think = ""
                    skeptic_think = ""
                    statistician_think = ""
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

        print("Description: ", description)
        print("Predictions: ", preds_real)

        # In LLM mode, any failure is a hard-stop (no static fallback).
        hard_stop = bool(use_llm and (not result.get("success", False)))

        if hard_stop or preds_real is None or test is None:
            smape_result = np.nan
            rmse_result = np.nan
            msmape_result = np.nan
            mae_result = np.nan
            mape_result = np.nan
            pocid_result = np.nan
            preds_real = []
            test_arr = np.array(test) if test is not None else np.array([])
        else:
            test_arr = np.array(test, dtype=float)
            preds_arr = np.array(preds_real, dtype=float)

            min_len = min(len(test_arr), len(preds_arr))
            if min_len == 0:
                smape_result = np.nan
                rmse_result = np.nan
                msmape_result = np.nan
                mae_result = np.nan
                mape_result = np.nan
                pocid_result = np.nan
                preds_real = []
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
            "predictions": [preds_real],
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
        }

        df_new = pd.DataFrame(data_serie)
        df_new = df_new.reindex(columns=COLS_SERIE)
        df_new.to_csv(path_csv, sep=";", mode="a", header=False, index=False)

        if hard_stop:
            raise RuntimeError(f"LLM run failed at dataset_index={i}. See CSV row description for details.")


if __name__ == "__main__":
    models = [
        "ARIMA",
        "ETS",
        "THETA",
        "svr",
        "rf",
        "catboost",
        "CWT_svr",
        "DWT_svr",
        "FT_svr",
        "CWT_rf",
        "DWT_rf",
        "FT_rf",
        "CWT_catboost",
        "DWT_catboost",
        "FT_catboost",
        "ONLY_CWT_catboost",
        "ONLY_CWT_rf",
        "ONLY_CWT_svr",
        "ONLY_DWT_catboost",
        "ONLY_DWT_rf",
        "ONLY_DWT_svr",
        "ONLY_FT_catboost",
        "ONLY_FT_rf",
        "ONLY_FT_svr",
        "NaiveSeasonal",
        "NaiveMovingAverage",
    ]

    exec_dataset_orchestrator(
        models,
        use_llm=True,
        ollama_model="mychen76/qwen3_cline_roocode:14b",
        debug=False,
        rolling="expanding",
        train_window=5,
        llm_logs=True,
        start_index=0,
        end_index=182,
    )
