"""Diagnostico das respostas vazias do gpt-oss no Ollama (rodar na maquina com GPU).

Monta o prompt real do turno 1 de uma serie (mesmo codigo do agente) e o envia direto
ao Ollama, sem langchain, variando `think` e `num_ctx`, N vezes cada. Para cada
chamada imprime: chars de content, chars de thinking, done_reason, eval_count e o
tamanho do prompt (prompt_eval_count). Uma resposta vazia com done_reason=length e
prompt_eval_count perto de num_ctx indica estouro de contexto; vazia com
done_reason=stop e thinking>0 indica que o modelo parou no canal de raciocinio.

Uso (todos os turnos): python outputs/resultados/scripts/diagnostico_ollama_vazio.py --think low --n 3 [--dataset M4_WEEKLY_DATASET --index 5]
Grade think x num_ctx so no turno 1: chamar main_grid() no final do arquivo.
"""
import argparse, json, os, sys
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ollama import Client
from orchestrator_react import ingest as I, pool as PM, react_loop as RL
from orchestrator_react.config import ReactConfig
from orchestrator_react.llm import ScriptedLLM, LLMError
from run_tsf_orchestrator import DEFAULT_MODELS
from truncamento_compact import ARTIFACTS, RESULTS_DIR


def real_prompt(ds, idx):
    art = json.load(open(os.path.join(ARTIFACTS, ds, f"dataset_{idx}.json"), encoding="utf-8"))
    cfg = ReactConfig.from_dict(art["config"])
    fr = I.load_dataset_frames(DEFAULT_MODELS, ds, RESULTS_DIR)
    ing = I.load_series(models=DEFAULT_MODELS, dataset=ds, dataset_index=idx, config=cfg,
                        results_dir=RESULTS_DIR, frames=fr, drop_models=[])
    st = ing.state
    st.strategy_prior = (art.get("cross_series") or {}).get("strategy_prior")
    PM.run_phase2(st, cfg)
    c = ScriptedLLM(responses=[])
    RL.run_react_loop(state=st, client=c, series_card=art["series_card"], pool_card=art["pool_card"],
                      config=cfg, diagnosis=art.get("diagnosis") or {})
    return c.calls[0]["system"], c.calls[0]["user"]


def all_prompts(ds, idx):
    """Prompts de todos os turnos (estado real, trajetoria gravada)."""
    from truncamento_compact import responses_from
    art = json.load(open(os.path.join(ARTIFACTS, ds, f"dataset_{idx}.json"), encoding="utf-8"))
    cfg = ReactConfig.from_dict(art["config"])
    fr = I.load_dataset_frames(DEFAULT_MODELS, ds, RESULTS_DIR)
    st = I.load_series(models=DEFAULT_MODELS, dataset=ds, dataset_index=idx, config=cfg,
                       results_dir=RESULTS_DIR, frames=fr, drop_models=[]).state
    st.strategy_prior = (art.get("cross_series") or {}).get("strategy_prior")
    PM.run_phase2(st, cfg)
    c = ScriptedLLM(responses=responses_from(art["react"]["trajectory"]))
    RL.run_react_loop(state=st, client=c, series_card=art["series_card"], pool_card=art["pool_card"],
                      config=cfg, diagnosis=art.get("diagnosis") or {})
    return [(x["system"], x["user"]) for x in c.calls]


def main():
    """Modo turnos: todos os turnos de uma serie, `think` dado, seeds 7..7+n-1.
    Para cada vazia, grava o thinking e diz se ele contem 'Action:'."""
    from orchestrator_react.llm import parse_agent_step
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-oss:20b"); ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--think", default="low", help="low|medium|high|none")
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--dataset", default="M4_WEEKLY_DATASET"); ap.add_argument("--index", type=int, default=5)
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    a = ap.parse_args()
    cli = Client(host=a.host, timeout=600)
    kw = {} if a.think == "none" else {"think": a.think}
    out_dir = os.path.join(REPO, "outputs", "diagnostico", "ollama_vazio"); os.makedirs(out_dir, exist_ok=True)
    print(f"think={a.think} num_ctx={a.num_ctx}: turno | seed | content | thinking | done | prompt_tok | eval | thinking tem 'Action:'?")
    for turn, (system, user) in enumerate(all_prompts(a.dataset, a.index), 1):
        for seed in range(7, 7 + a.n):
            r = cli.chat(model=a.model, stream=False, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                         options={"temperature": 0.2, "seed": seed, "num_ctx": a.num_ctx}, **kw)
            m = r["message"]; c, t = m.get("content") or "", m.get("thinking") or ""
            has_action = bool(t) and parse_agent_step(t).ok
            print(f"{turn:>2} | {seed} | {len(c)} | {len(t)} | {r.get('done_reason')} | {r.get('prompt_eval_count')} | {r.get('eval_count')} | {has_action if not c else '-'}", flush=True)
            if not c:
                open(os.path.join(out_dir, f"turno{turn}_seed{seed}_{a.think}.txt"), "w").write(t)


def main_grid():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-oss:20b"); ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--dataset", default="M4_WEEKLY_DATASET"); ap.add_argument("--index", type=int, default=5)
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    a = ap.parse_args()
    system, user = real_prompt(a.dataset, a.index)
    cli = Client(host=a.host, timeout=600)
    print(f"prompt: {len(system)+len(user)} chars")
    print("think | num_ctx | vazias/N | content_chars | thinking_chars | done_reason | prompt_tokens | eval_count")
    for think in (None, "low", "medium", False):
        for num_ctx in (8192, 16384):
            rows = []
            for _ in range(a.n):
                kw = {} if think is None else {"think": think}
                r = cli.chat(model=a.model, stream=False, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                             options={"temperature": 0.2, "seed": 7, "num_ctx": num_ctx}, **kw)
                m = r["message"]
                rows.append((len(m.get("content") or ""), len(m.get("thinking") or ""), r.get("done_reason"), r.get("prompt_eval_count"), r.get("eval_count")))
            empty = sum(1 for x in rows if x[0] == 0)
            print(f"{think!s:6}| {num_ctx} | {empty}/{a.n} | {[x[0] for x in rows]} | {[x[1] for x in rows]} | {[x[2] for x in rows]} | {rows[0][3]} | {[x[4] for x in rows]}", flush=True)


if __name__ == "__main__":
    main()  # trocar por main_grid() para a grade think x num_ctx do turno 1
