"""Valida a instrumentacao de tempo/registro por passo (react_loop.py, llm.py).

Sem GPU: usa o mesmo replay do truncamento_compact.py (ScriptedLLM com a trajetoria
gravada), mas com um cliente que dorme um tempo fixo por chamada e expoe `last_meta`
no formato do Ollama. As ferramentas rodam de verdade, entao `tool_exec_s` e real.
Nada aqui mede o tempo do LLM real.

Saida: outputs/resultados/ablacao/instrumentacao_validacao.csv (+ resumo em stdout)
"""
import json, os, sys, time
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from orchestrator_react import ingest as ingest_mod, pool as pool_mod, react_loop as RL
from orchestrator_react.config import ReactConfig
from orchestrator_react.llm import ScriptedLLM, LLMError
from run_tsf_orchestrator import DEFAULT_MODELS
import common as C
from truncamento_compact import responses_from, ARTIFACTS, RESULTS_DIR

SLEEP = 0.05


class TimedScripted(ScriptedLLM):
    last_meta: dict = None

    def complete(self, system, user):
        self.last_meta = {}
        time.sleep(SLEEP)
        out = super().complete(system, user)
        self.last_meta = {"prompt_eval_count": len(user) // 4, "eval_count": len(out) // 4,
                          "eval_duration": int(SLEEP * 1e9), "done_reason": "stop"}
        return out


def main():
    sample = [("NN5_WEEKLY_DATASET", i) for i in range(8)] + [("ETTM2", i) for i in range(7)]
    rows, frames, drops = [], {}, {}
    for ds, idx in sample:
        art = json.load(open(os.path.join(ARTIFACTS, ds, f"dataset_{idx}.json"), encoding="utf-8"))
        cfg = ReactConfig.from_dict(art["config"])
        if ds not in frames:
            frames[ds] = ingest_mod.load_dataset_frames(DEFAULT_MODELS, ds, RESULTS_DIR)
            try:
                drops[ds] = sorted(ingest_mod.find_misaligned_models(
                    DEFAULT_MODELS, ds, 0, results_dir=RESULTS_DIR, frames=frames[ds],
                    n_windows=cfg.n_validation_windows))
            except Exception:
                drops[ds] = []
        t0 = time.perf_counter()
        ing = ingest_mod.load_series(models=DEFAULT_MODELS, dataset=ds, dataset_index=idx, config=cfg,
                                     results_dir=RESULTS_DIR, frames=frames[ds], drop_models=drops[ds])
        state = ing.state
        state.strategy_prior = (art.get("cross_series") or {}).get("strategy_prior")
        pool_mod.run_phase2(state, cfg)
        setup = time.perf_counter() - t0
        traj = art["react"]["trajectory"]
        res = RL.run_react_loop(state=state, client=TimedScripted(responses=responses_from(traj)),
                                series_card=art["series_card"], pool_card=art["pool_card"],
                                config=cfg, diagnosis=art.get("diagnosis") or {})
        llm = sum(e["llm_call_s"] for e in res.trajectory) + res.llm_failed_turn_s; tool = sum(e["tool_exec_s"] for e in res.trajectory)
        light = json.dumps({"trajectory": res.trajectory}, ensure_ascii=False)
        full = json.dumps({"trajectory": res.trajectory, "step_details": res.step_details}, ensure_ascii=False)
        rows.append(dict(dataset=ds, idx=idx, steps=len(res.trajectory), details=len(res.step_details),
                         elapsed_s=res.elapsed_s, llm_s=llm, tool_s=tool, resid_s=res.elapsed_s - llm - tool,
                         setup_s=setup, stop=res.stop_reason, bytes_traj=len(light), bytes_traj_details=len(full),
                         has_keys=all({"llm_call_s", "tool_exec_s"} <= set(e) for e in res.trajectory),
                         valid=bool(json.loads(full))))
    d = pd.DataFrame(rows); d.to_csv(os.path.join(C.OUTPUTS_BASE, "ablacao", "instrumentacao_validacao.csv"), index=False)
    pd.set_option("display.width", 200); print(d.round(3).to_string())
    print(d[["elapsed_s", "llm_s", "tool_s", "resid_s", "setup_s"]].sum().round(3).to_dict())


if __name__ == "__main__":
    main()
