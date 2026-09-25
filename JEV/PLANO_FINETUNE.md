# Plano — Fine-tune do LAYA como GATE aprendido de transferência

> **Objetivo:** treinar um classificador local (LAYA fine-tuned) que, dada uma
> estratégia candidata + estado da série, prevê **se ela vai bater o piso das
> sementes na janela de teste** — e usar essa previsão como gate na decisão final.
>
> **Por quê:** medimos que (a) a decisão final é o gargalo (transferência
> validação→teste de 10% no LLM e 14,5% no LAYA zero-shot), (b) o classificador
> zero-shot ≈ piso em todos os 7 datasets, (c) estado RICO (multilingual 8192)
> importa. O fine-tune é o único caminho que muda a função-objetivo de "ganhar
> nas 3 janelas" para "transferir pro teste" — porque o RÓTULO vem do desfecho
> real das séries passadas.

**Arquitetura:** pipeline em 4 estágios, todos determinísticos exceto o treino:
(1) export do dataset de decisões rotuladas por replay dos runs existentes;
(2) baseline de regressão logística (saber se há sinal antes de gastar GPU);
(3) fine-tune do LAYA na 5090 (recipe oficial RLCD); (4) gate integrado + avaliação.

**Tech stack:** python + `orchestrator_react` (replay), `laya` (fine-tune/inferência),
`scikit-learn` (baseline), pandas/numpy.

**Hardware:** treino na RTX 5090 do servidor (32 GB — folga). Export e baseline
rodam em CPU local.

**Dados (todos já existem em disco):**
- `orchestrator_react_v5/llm_artifacts/NN5_WEEKLY_DATASET/` — 456 propostas do LLM
- `orchestrator_laya_laya_v0/*.csv` — traces do LAYA em 7 datasets
- `orchestrator_laya_laya_v0_ml8192/` e `_noseeds/` — traces adicionais
- CSVs dos modelos base + `.tsf` (replay determinístico dos desfechos)

---

## Global Constraints

- **Anti-vazamento**: treinar e avaliar com **leave-one-dataset-out** — o gate
  que decide a série *i* nunca viu o desfecho da série *i* (nem de séries do
  mesmo dataset no fold de avaliação).
- **Rótulo é desfecho, não imitação**: `label = (smape_teste < melhor_semente_teste)`.
- **Estado é validação-only**: o input do classificador é exatamente o que o
  agente vivo vê (cards + histórico), nunca a janela de teste.
- Nada de chamadas ao LLM em nenhum estágio (o treino não precisa; os dados vêm
  de replay).

---

## File Structure

| arquivo | responsabilidade |
|---|---|
| `JEV/build_finetune_dataset.py` | replay determinístico → `JEV/data/gate_dataset.jsonl` (uma linha por estratégia avaliada) |
| `JEV/run_gate_baseline.py` | regressão logística sobre features tabulares + métricas (AUC/Brier) por dataset LOO |
| `JEV/finetune_laya_gate.py` | treino RLCD do LAYA sobre o dataset exportado (roda na 5090) |
| `JEV/eval_gate.py` | avalia checkpoint fine-tuned: ranking por P(transferir), sweep de limiar, sMAPE final vs piso |
| `JEV/PLANO_FINETUNE.md` | este plano |

---

### Task 1: Export do dataset de decisões rotuladas

**Files:**
- Create: `JEV/build_finetune_dataset.py`

**Interfaces:**
- Produz: `JEV/data/gate_dataset.jsonl`, linhas:
  ```json
  {"source": "gpt_oss_v5", "dataset": "NN5_WEEKLY_DATASET", "series": 0,
   "state": "<texto rico: cards + histórico + features numéricas>",
   "spec": {"combine": "...", "pool": "...", ...},
   "score_val": 0.7048, "smape_test": 0.0418, "floor_smape_test": 0.0422,
   "label": 1, "origin": "agent", "turn": 3,
   "rank": 2, "margem_pct": 0.031, "n_attempts": 11, "tau": 0.42, "n_models": 19}
  ```

- [ ] **Step 1: Replay de uma fonte (LLM v5 NN5)** — reusar a lógica de replay do
`diagnostics/analyze_run.py`: carregar série, `run_phase2`, aplicar cada passo do
`react.trajectory` via `registry.call_tool`, e para cada `evaluate_strategy`
capturar `(spec, score_val, smape_test, floor, turn)`.

```python
# JEV/build_finetune_dataset.py (esqueleto do núcleo)
for idx in range(n_series):
    ing = I.load_series(models=MODELS, dataset=DS, dataset_index=idx, config=cfg,
                        results_dir=BASE, source_file=TSF, source_dir=SRC, frames=frames)
    st = ing.state
    POOL.run_phase2(st, cfg)
    floor = min(smape(st.apply_to_test(a.spec)[0], ing.test_values)
                for a in st.attempts if a.origin == "baseline")
    for a in st.attempts:                      # sementes também são exemplos
        emit(example(st, a, floor, ing, turn=0))
    for entry in art["react"]["trajectory"]:
        if entry["action"] != "evaluate_strategy":
            call_tool(st, entry["action"], dict(entry.get("action_args") or {}), withheld={})
            continue
        ok, _ = call_tool(st, "evaluate_strategy", dict(entry.get("action_args") or {}), withheld={})
        if ok:
            emit(example(st, st.attempts[-1], floor, ing, turn=entry["iteration"]))
```

- [ ] **Step 2: Replay das fontes LAYA** — os CSVs `orchestrator_laya_*` têm
`description.loop.trace` com as ações; mapear `label → spec` (função
`label_to_spec` já validada) e refazer `state.evaluate` na ordem, registrando o
mesmo exemplo. Pools reais são reconstruídos por `run_phase2` (mesma ordem).

- [ ] **Step 3: Estado rico** — `state` do exemplo usa `build_state_text` SEM
compressão (budget 8000) + features numéricas explícitas em JSON
(`score_val`, `rank`, `margem_pct` vs líder, `n_attempts`, `tau` da
estabilidade de ranking, `n_models`, `origin`). Esses campos entram na linha
JSONL além do texto do estado — o Task 2 os usa como features tabulares.

- [ ] **Step 4: Rótulo + hold-out** — `label = int(smape_test < floor_smape_test)`;
gravar `dataset` e `series` para o split LOO do Task 2/4.

- [ ] **Step 5: Rodar e conferir contagem** — `python JEV/build_finetune_dataset.py`
imprime `{fonte: n_exemplos}`. Esperado: ~450 (LLM NN5) + ~300 (LAYA NN5) +
~centenas (LAYA 7 datasets) ≈ **1.000-1.500 exemplos**.

---

### Task 2: Baseline de regressão logística (sem GPU)

**Files:**
- Create: `JEV/run_gate_baseline.py`

**Interfaces:**
- Consome: `JEV/data/gate_dataset.jsonl`
- Produz: tabela por dataset (fold LOO): AUC, Brier, `n`, taxa base de `label=1`

- [ ] **Step 1: Features tabulares** — por exemplo: `score_val`, `rank`,
`margem_pct vs líder`, `origin`, `n_models`, `tau` (estabilidade), `n_attempts`.
- [ ] **Step 2: LOO por dataset** — treinar nos 6 datasets, avaliar no 7º,
reportar AUC/Brier/taxa-base por fold.
- [ ] **Step 3: Decisão de portão** — se AUC ≤ 0.55 em todos os folds → o sinal
não existe nos features atuais → parar e revisar o desenho ANTES de gastar GPU;
se AUC ≥ 0.6 em algum fold → seguir para o Task 3.

---

### Task 3: Fine-tune do LAYA (5090)

**Files:**
- Create: `JEV/finetune_laya_gate.py`

**Interfaces:**
- Consome: `JEV/data/gate_dataset.jsonl` (split treino/val LOO)
- Produz: `JEV/models/laya_gate_<fold>/` (checkpoint fine-tuned local)

- [ ] **Step 1: Formato do dataset de decisões do LAYA** — uma decisão =
`state` (texto rico) + question `noul`:
  ```python
  question = {"transfer": {"type": "noul",
      "instructions": "Will this strategy beat the best seeded baseline on the blind test window?",
      "criteria": ["it will be worse or tied", "it will be strictly better"]}}
  ```
  resposta correta = `label`. O notebook oficial (`laya_finetune_typed_decisions_2xT4_kaggle.ipynb`)
  serve de base: dataset → treino RLCD (GRPO, `laya.proper_reward`) → ajuste de
  temperaturas → avaliação.
- [ ] **Step 2: Treinar 1 fold piloto** (ex.: avaliar em ETTM2) — ~10 min na 5090
  para ~1k exemplos × 4 épocas (referência: 4-5 h para 30k perguntas em 2×T4).
- [ ] **Step 3: Salvar checkpoint local** — diretório com `model.safetensors` +
  `rl_agent_config.json` (o `laya_loop.LayaAgent` já carrega caminho local).

---

### Task 4: Avaliação do gate + integração

**Files:**
- Create: `JEV/eval_gate.py`

**Interfaces:**
- Consome: `JEV/models/laya_gate_<fold>/` + `JEV/data/gate_dataset.jsonl`
- Produz: por fold — AUC/Brier da P(transferir), e **sMAPE final simulada**:
  para cada série do fold, aplicar o gate (só trocar a semente quando
  P > limiar) e comparar com o piso e com o run original.

- [ ] **Step 1: Métricas do gate** — P(transferir) rankeia estratégias? (AUC por fold)
- [ ] **Step 2: Sweep de limiar** — para τ ∈ {0.3…0.9}, simular a Fase 4:
  `estratégia = melhor semente` a menos que alguma proposta tenha P > τ;
  reportar sMAPE final vs piso por dataset. **Critério de sucesso: bater o piso
  (0.1156 NN5 / 0.2030 ETTM2) e o run gpt-oss (0.1177 / 0.1610) sem piorar nenhum dataset.**
- [ ] **Step 3: Integrar no loop** — se o gate vencer, plugar como guarda na
  Fase 4 do `run_tsf_orchestrator.py` (ou no `laya_loop`): consulta barata
  (~35 ms GPU / 200-500 ms CPU) antes de aceitar um override de semente.

---

## Riscos e decisões em aberto

1. **Rótulo ruidoso**: o desfecho no teste é UMA janela — alta variância. Mitigação:
   só decisões binárias (melhor/pior que o piso, não magnitude) + LOO.
2. **Tamanho da amostra** (~1k): se o baseline logístico não achar sinal, revisar
   features antes do fine-tune (Task 2 é o funil).
3. **O gate é ortogonal ao LLM**: não substitui o agente — decide se a proposta
   dele vale. Pode ser combinado com o gate estatístico (verdict) como feature.
4. Se o fine-tune do LAYA travar em dependências (transformers/torch da 5090),
   fallback: fine-tune ModernBERT-base com `Trainer` do transformers (mesmo dado,
   head binária) — mesmo pipeline, 1 h extra de código.
