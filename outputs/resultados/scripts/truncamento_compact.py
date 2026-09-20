"""Frequencia do truncamento em `_compact` (orchestrator_react/prompts.py).

Para cada serie da amostra:
  1. reconstroi o estado real (ingestao dos modelos + Fase 2: sementes) -- sem o
     meta-modelo pooled (treinaria no dataset inteiro) e com o combine_dba atual;
  2. usa o series_card / pool_card / diagnosis / strategy_prior GRAVADOS no artefato
     de orchestrator_react_v5 (os mesmos que o agente viu);
  3. reexecuta `run_react_loop` com um LLM roteirizado (`ScriptedLLM`) que repete a
     trajetoria gravada (Thought/Action/Action Input de cada passo), de modo que
     `build_turn_prompt` real roda com scratchpad e observacoes reais;
  4. um wrapper em `prompts._compact` registra cada chamada: rotulo (pelo ponto de
     chamada), limite, tamanho do JSON, se dispara corte, e o resultado da versao
     ANTIGA (embutida aqui) e da versao que esta em prompts.py no momento da execucao.

Uso:  python outputs/resultados/scripts/truncamento_compact.py <tag> [--limit N]
Saida: outputs/resultados/ablacao/truncamento_chamadas_<tag>.csv
"""
import argparse
import json
import os
import random
import re
import sys
import time
import traceback

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from orchestrator_react import ingest as ingest_mod
from orchestrator_react import pool as pool_mod
from orchestrator_react import prompts as P
from orchestrator_react import react_loop as RL
from orchestrator_react.config import ReactConfig
from orchestrator_react.llm import ScriptedLLM
from run_tsf_orchestrator import DEFAULT_MODELS

import common as C

RESULTS_DIR = os.path.join(REPO, "timeseries", "mestrado", "resultados")
ARTIFACTS = os.path.join(RESULTS_DIR, "orchestrator_react_v5", "llm_artifacts")
N_PER_LARGE = 25  # ANP, NN5, M4; os 4 ETT entram inteiros (7 series cada)

_PROMPTS_SRC = open(P.__file__, encoding="utf-8").read().splitlines()
_ORIG_COMPACT = P._compact


def old_compact(payload, limit=1800):
    """Implementacao antiga (antes da correcao), embutida para comparacao."""
    text = json.dumps(payload, ensure_ascii=False, default=str, separators=(",", ":"))
    return text if len(text) <= limit else text[:limit] + " ...[truncated]"


def label_for(lineno: int) -> str:
    line = _PROMPTS_SRC[lineno - 1]
    if "_slim_series_card" in line:
        return "series_card"
    if "_slim_pool_card" in line:
        return "pool_card"
    if "limit=900" in line:
        return "dataset_card"
    if "_slim_diagnosis" in line:
        return "diagnosis"
    if "a.brief" in line:
        return "historico"
    if "handles, limit=600" in line:
        return "handles"
    if "entry['action_args']" in line:
        return "scratchpad_args"
    if "last_observation" in line:
        return "ultima_observacao"
    if "observation, limit=160" in line:
        return "resumo_observacao"
    return f"linha_{lineno}"


def cut_kind(text: str, limit: int) -> str:
    prefix = text[:limit]
    in_string = False
    esc = False
    for ch in prefix:
        if esc:
            esc = False
        elif ch == "\\":
            esc = True
        elif ch == '"':
            in_string = not in_string
    if in_string:
        return "dentro_de_string"
    last, nxt = prefix[-1], text[limit]
    num = set("0123456789.-+eE")
    if last in num and nxt in num:
        return "dentro_de_numero"
    if last.isalpha() and nxt.isalpha():
        return "dentro_de_literal"
    return "estrutural"


def parses(s: str) -> bool:
    try:
        json.loads(s)
        return True
    except Exception:
        return False


CTX = {"dataset": "", "idx": -1, "iteration": 0}
RECORDS = []


def wrapper(payload, limit=1800):
    lineno = sys._getframe(1).f_lineno
    label = label_for(lineno)
    text = json.dumps(payload, ensure_ascii=False, default=str, separators=(",", ":"))
    triggered = len(text) > limit
    rec = {
        "dataset": CTX["dataset"], "dataset_index": CTX["idx"], "iteration": CTX["iteration"],
        "tipo": label, "limite": limit, "tamanho": len(text), "dispara": triggered,
    }
    cur = _ORIG_COMPACT(payload, limit)
    if triggered:
        old_out = old_compact(payload, limit)
        cut = old_out[: -len(" ...[truncated]")]
        rec["antigo_json_valido"] = parses(cut)
        rec["antigo_corte"] = cut_kind(text, limit)
        marker = " [truncated: some fields omitted]"
        new_json = cur[: -len(marker)] if cur.endswith(marker) else cur
        rec["atual_json_valido"] = parses(new_json)
        rec["atual_string_inteira_valida"] = parses(cur)
        rec["atual_cabe_no_limite"] = len(new_json) <= limit
        rec["atual_fallback_nota"] = new_json.startswith('{"note":"payload too large')
        rec["atual_tem_marcador_antigo"] = cur.endswith("...[truncated]")
    RECORDS.append(rec)
    return cur


_orig_build_turn = P.build_turn_prompt


def build_turn_wrapper(*args, **kwargs):
    CTX["iteration"] = kwargs.get("iteration", CTX["iteration"])
    return _orig_build_turn(*args, **kwargs)


def responses_from(trajectory):
    out = []
    for step in trajectory:
        action = step.get("action") or ""
        if action in ("", "unparsed"):
            continue
        out.append(
            f"Thought: {step.get('thought') or ''}\nAction: {action}\n"
            f"Action Input: {json.dumps(step.get('action_args') or {}, ensure_ascii=False)}"
        )
    return out


def pick_sample(seed=0):
    rng = random.Random(seed)
    sample = []
    for ds in C.DATASETS:
        files = sorted(os.listdir(os.path.join(ARTIFACTS, ds)), key=lambda f: int(re.findall(r"\d+", f)[0]))
        idxs = [int(re.findall(r"\d+", f)[0]) for f in files]
        chosen = idxs if len(idxs) <= 7 else sorted(rng.sample(idxs, N_PER_LARGE))
        sample += [(ds, i) for i in chosen]
    return sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    P._compact = wrapper
    P.build_turn_prompt = build_turn_wrapper
    RL.P = P  # o loop chama P.build_turn_prompt

    sample = pick_sample()
    if args.limit:
        sample = sample[: args.limit]
    print(f"amostra: {len(sample)} series", flush=True)

    frames_by_ds, drop_by_ds = {}, {}
    status = []
    t0 = time.time()
    for n, (ds, idx) in enumerate(sample, 1):
        try:
            art = json.load(open(os.path.join(ARTIFACTS, ds, f"dataset_{idx}.json"), encoding="utf-8"))
            cfg = ReactConfig.from_dict(art["config"])
            if ds not in frames_by_ds:
                frames_by_ds[ds] = ingest_mod.load_dataset_frames(DEFAULT_MODELS, ds, RESULTS_DIR)
                try:
                    bad = ingest_mod.find_misaligned_models(
                        DEFAULT_MODELS, ds, 0, results_dir=RESULTS_DIR, frames=frames_by_ds[ds],
                        n_windows=cfg.n_validation_windows,
                    )
                except Exception:
                    bad = {}
                drop_by_ds[ds] = sorted(bad)
            ing = ingest_mod.load_series(
                models=DEFAULT_MODELS, dataset=ds, dataset_index=idx, config=cfg,
                results_dir=RESULTS_DIR, frames=frames_by_ds[ds], drop_models=drop_by_ds[ds],
            )
            state = ing.state
            state.strategy_prior = (art.get("cross_series") or {}).get("strategy_prior")
            pool_mod.run_phase2(state, cfg)

            CTX.update(dataset=ds, idx=idx, iteration=0)
            traj = art["react"]["trajectory"]
            client = ScriptedLLM(responses=responses_from(traj))
            res = RL.run_react_loop(
                state=state, client=client, series_card=art["series_card"], pool_card=art["pool_card"],
                config=cfg, diagnosis=art.get("diagnosis") or {},
            )
            status.append({"dataset": ds, "dataset_index": idx, "ok": True,
                           "passos_gravados": len(traj), "passos_reexecutados": len(res.trajectory),
                           "stop": res.stop_reason})
        except Exception as exc:
            status.append({"dataset": ds, "dataset_index": idx, "ok": False, "erro": f"{type(exc).__name__}: {exc}"})
            traceback.print_exc()
        if n % 10 == 0 or n == len(sample):
            print(f"[{n}/{len(sample)}] {time.time() - t0:.0f}s", flush=True)

    out_dir = os.path.join(C.OUTPUTS_BASE, "ablacao")
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(RECORDS).to_csv(os.path.join(out_dir, f"truncamento_chamadas_{args.tag}.csv"), index=False)
    pd.DataFrame(status).to_csv(os.path.join(out_dir, f"truncamento_replay_{args.tag}.csv"), index=False)
    print("ok:", sum(s["ok"] for s in status), "falhas:", sum(not s["ok"] for s in status))


if __name__ == "__main__":
    main()
