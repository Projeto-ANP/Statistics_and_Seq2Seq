# Orchestrator Pipeline (determinístico + LLM opcional)

Este documento explica como o pipeline do `orchestrator/` funciona, como os dados são interpretados e como alternar entre:

- **Modo 1 (Determinístico)**: sem LLM, 100% código.
- **Modo 2 (LLM + avaliação determinística)**: LLM faz escolhas **controladas porém exploratórias** (seleciona um subset inicial, ajusta parâmetros e pode **adicionar** candidatos reais de um universo/whitelist durante o debate), mas **quem decide é o avaliador determinístico**.

> Objetivo: combinar previsões multi-step (H passos à frente) de vários modelos e escolher a melhor estratégia com base em backtest/validação por janelas.

---

## 1) Contrato de dados (como o sistema “entende” os dados)

O pipeline usa o mesmo `context` do projeto (ver `agent/context.py`). Ele precisa de:

### 1.1 `CONTEXT_MEMORY["all_validations"]`
Estrutura esperada:

```python
{
  "predictions": [
    {"ARIMA": [h1,h2,...], "ETS": [...], ...},  # window 0
    {"ARIMA": [h1,h2,...], "ETS": [...], ...},  # window 1
    ...
  ],
  "test": [
    [y1,y2,...],  # window 0
    [y1,y2,...],  # window 1
    ...
  ]
}
```

- Cada **window** é uma “origem” (um ponto no tempo onde você prevê H passos à frente).
- Em cada window, cada modelo fornece uma lista `[h1, h2, ..., hH]`.

### 1.2 `CONTEXT_MEMORY["predictions"]` (para o teste final)
Estrutura:

```python
{
  "ARIMA": [h1,h2,...],
  "ETS":  [h1,h2,...],
  ...
}
```

Isso representa as previsões do **final_test** (a janela final) de cada modelo. O pipeline aplica a estratégia vencedora (aprendida na validação) para gerar a previsão combinada final.

### 1.3 Normalização interna
O loader do `orchestrator` converte o contrato acima para matrizes NumPy:

- `y_true`: shape `(n_windows, H)`
- `y_preds`: shape `(n_windows, n_models, H)`

Se houver diferenças de tamanho (por exemplo, algum modelo com menos passos), o loader **trunca para o menor horizonte consistente**.

---

## 2) Métricas (como a avaliação é feita)

O pipeline calcula:

- **MAPE**, **sMAPE**, **RMSE**
- **POCID** (direcional)

E reporta:

- métricas **por horizonte** (h=1..H)
- métricas **agregadas** (média sobre os horizontes)
- **estabilidade** (desvio padrão das métricas entre janelas)

### Observação sobre MAPE e zeros
MAPE pode explodir quando `y_true=0`. Por padrão o pipeline usa um tratamento “safe” (pode ignorar pontos com denominador ~0, dependendo da config).

---

## 3) Anti-leakage (regra central)

As estratégias que **selecionam** modelos ou **aprendem pesos** precisam ser avaliadas de forma que não usem a window atual para escolher os pesos da própria window.

No `orchestrator`, isso é garantido assim:

- Para a window `i`, a estratégia usa **somente janelas `< i`** para selecionar/ajustar.
- Isso pode ser:
  - **expanding**: treina com `0..i-1`
  - **rolling**: treina com um histórico fixo (ex.: últimas `train_window` janelas)

Quando `i=0` (sem histórico), o pipeline usa um fallback (ex.: média) para não inventar nada.

---

## 4) Estratégias disponíveis (toolbox determinístico)

As estratégias são selecionadas via `params.method`.

### 4.1 Baselines (não aprendem)
- `mean`: média simples entre modelos, por horizonte.
- `median`: mediana entre modelos, por horizonte (robusta a outliers).
- `trimmed_mean`: remove extremos e faz média (robusta). Parâmetro:
  - `trim_ratio` (ex.: 0.2)

### 4.2 Seleção (aprende por janela usando apenas passado)
- `best_single`: escolhe 1 modelo global (menor RMSE histórico agregado).
- `best_per_horizon`: escolhe o melhor modelo para cada horizonte `h`.
- `topk_mean_per_horizon`: escolhe top-k por horizonte e faz média. Parâmetro:
  - `top_k` (ex.: 3)

### 4.3 Pesos / stacking (aprendem pesos por horizonte)
- `inverse_rmse_weights_per_horizon`:
  - pesos `w(h) ∝ 1/RMSE(h)`
  - pode limitar a `top_k`
  - pode aplicar `shrinkage` (encolher para pesos uniformes)

- `ridge_stacking_per_horizon`:
  - ridge por horizonte
  - projeta pesos para o simplex (`w>=0`, `sum(w)=1`) para estabilidade
  - parâmetros: `l2` (regularização) e opcionalmente `top_k`

---

## 5) Como o Modo 1 (determinístico) funciona

Fluxo (alto nível):

1. Carrega `all_validations` do context
2. Para cada estratégia candidata:
   - Gera previsões combinadas por window/horizonte seguindo anti-leakage
   - Calcula métricas por horizonte + agregado + estabilidade
3. Compara todas e escolhe um vencedor por **score composto**
4. Aplica o vencedor em `context["predictions"]` (final_test) e devolve a previsão final

### Score composto (para ranking)
O pipeline usa um score multi-objetivo normalizado por um baseline (média):

- Importante: o avaliador calcula `baseline_mean` **internamente** para normalização, mesmo que `baseline_mean` não esteja na lista de candidatos escolhidos/debatidos.

- Normalização típica:
  - `RMSE_norm = RMSE / RMSE_baseline_mean`
  - idem para `sMAPE` e `MAPE`

Score (quanto menor, melhor):

```text
score = a*RMSE_norm + b*sMAPE_norm + c*MAPE_norm - d*(POCID/100)
```

---

## 6) Como o Modo 2 (LLM + avaliação determinística) funciona

**Pipeline (modo LLM) – passo a passo, simples e intuitivo**

- **Início**
  - **Entrada (por dataset_index)**:
    - `all_validations`: janelas de validação com:
      - `y_true` (o “test” de cada janela)
      - `preds` de **vários modelos** para cada janela (multi-step)
    - `predictions`: as **previsões finais** (no `final_test`) desses mesmos modelos
    - `models_available`: lista de modelos disponíveis
  - Isso é preparado no runner chamando `init_context()` + `generate_all_validations_context(models, dataset_index)`.

- **Etapa 1 — Proposer (LLM, mas “travado” em 1 tool)**
  - **Objetivo**: escolher um subconjunto de estratégias candidatas *condicionado aos dados*.
  - **Como funciona de verdade**:
    - O Proposer é obrigado a chamar `proposer_brief_tool()`.
    - Essa tool é **determinística** e lê `all_validations` para construir um *brief* com:
      - `validation_summary` (estatísticas reais da validação)
      - `candidate_library` (um **universo/whitelist** maior de candidatos válidos)
      - `recommended_candidates` (um shortlist determinístico sugerido, só para guiar)
      - `score_presets` (presets de pesos do score)
    - **A LLM então toma decisões** (JSON) usando esse brief:
      - `selected_names`: quais candidatos do `candidate_library` entram
      - `params_overrides`: ajustes limitados (somente `top_k`, `trim_ratio`, `shrinkage`, `l2`)
      - `score_preset`: qual objetivo priorizar (ex.: `balanced`, `rmse_focus`, `direction_focus`)
      - `force_debate` e `debate_margin`
  - **Importante (robustez)**: a tool não recebe parâmetros da LLM; o pipeline grava os inputs no contexto via `set_context(...)` e a tool lê via `get_context(...)`.

- **Etapa 2 — Debate (opcional, com ações limitadas e evidência real)**
  - **Objetivo**: permitir interpretação/ajuste sem inventar números e sem “trocar o método”.
  - **Quando acontece**:
    - `debate=True` força.
    - ou `force_debate=True` vindo do Proposer.
    - ou `debate_auto=True` ativa se top-1 vs top-2 estiverem muito próximos (margem < um limiar efetivo).
      - O limiar efetivo nunca é reduzido pelo Proposer (para não “desligar” debate sem querer).
  - **O que Skeptic/Statistician podem fazer**:
    - somente:
      - adicionar candidatos reais do universo (`add_names`), e/ou
      - remover candidatos (`remove_names`), e/ou
      - ajustar **apenas** `top_k`, `trim_ratio`, `shrinkage`, `l2`
    - não podem:
      - criar candidato novo (fora do universo)
      - trocar `params.method`
      - citar números que não vieram da tool
  - **Como eles se baseiam em dados reais**:
    - são obrigados a chamar `build_debate_packet_tool()`.
    - essa tool devolve um pacote determinístico com ranking real, margem top2, winners por horizonte, estabilidade etc.
    - e também inclui um bloco `universe` com leaderboards (inclui top por **POCID/direção**) para a LLM poder **adicionar** candidatos úteis.
  - **Proteções extras contra alucinação**:
    - o prompt inclui `valid_candidate_names: [...]` (universo) e `current_candidate_names: [...]` (subset atual).
    - o pipeline valida o JSON: se `add_names` / `remove_names` / `params_overrides` tiverem nomes que não existem no universo, dá **hard-stop**.

  > Observação prática: `selected_names` é o JSON bruto do Proposer (o que ele “pediu”).
  > O conjunto **real** de candidatos usado após cada etapa é rastreado no `description` (campo `candidates_trace`) e nas colunas do CSV (`final_candidate_names`).
  > Se o Proposer selecionar algum nome que não exista no universo/whitelist, isso aparece em `candidates_trace.dropped_selected_names`.

- **Etapa 3 — Avaliação final (100% determinística, anti-leakage)**
  - **Objetivo**: escolher o melhor candidato de forma verificável.
  - **Como decide**:
    - roda `evaluate_all(...)` em código, com validação rolling/expanding sem leakage.
    - computa RMSE, sMAPE, MAPE, POCID (agregado + por horizonte), estabilidade, e score normalizado vs `baseline_mean`.
  - Auditoria:
    - `context["orchestrator_last_eval"]` guarda o resultado.
    - `context["tools_called"]` registra as tools chamadas pelos agentes.
    - por compatibilidade histórica, o pipeline também adiciona o marcador `evaluate_strategies_tool` em `tools_called`.

- **Etapa 4 — Predição final (determinística) + rastreio de pesos/modelos**
  - **Objetivo**: gerar o forecast final multi-step para o `final_test`.
  - **Como funciona**:
    - aplica o `best_candidate` sobre `context["predictions"]`.
    - retorna `predict_debug` com o “como” da combinação, por exemplo:
      - modelos escolhidos (best_single / best_per_horizon / top-k)
      - `weights_by_horizon` quando o método aprende pesos (inverse-RMSE / ridge)

- **Etapa 5 — Salvamento no CSV (por dataset_index)**
  - O runner salva:
    - métricas no `final_test` (mape/smape/rmse/pocid etc.)
    - `description` (JSON completo de auditoria)
    - colunas explícitas para facilitar rastreio (ver seção 9.1)

### 6.1 Por que tools sem parâmetros?

Alguns modelos pequenos erram ao passar argumentos em tool calls, mas acertam em **chamar** a tool.

Para robustez:

- o pipeline grava os inputs no contexto (`set_context`) antes de cada agente rodar
- as tools leem esses inputs via `get_context`
- quaisquer argumentos passados pela LLM são ignorados/preteridos quando o contexto já está preenchido

---

## 7) Onde alternar entre Modo 1 e Modo 2

O switch foi exposto no runner no mesmo estilo do seu `run_tsf_agents.py`:

- `exec_dataset_orchestrator(models, use_llm=False, ...)` → **Modo 1**
- `exec_dataset_orchestrator(models, use_llm=True, ...)` → **Modo 2**

No modo 2 você também passa:

- `ollama_model` (ex.: `mychen76/qwen3_cline_roocode:4b`)
- `rolling` e `train_window`

---

## 8) Exemplos de execução

### 8.1 Executar o loop por dataset (Modo 1 – determinístico)

```bash
conda activate agno
cd /home/anp/Documents/lucas_mestrado/Statistics_and_Seq2Seq
python run_tsf_orchestrator.py
```

Por padrão, o script está configurado para `use_llm=False`.

### 8.2 Executar o loop por dataset (Modo 2 – com LLM)

Edite a chamada no final do `run_tsf_orchestrator.py` e use:

```python
exec_dataset_orchestrator(
    models,
    use_llm=True,
    ollama_model="mychen76/qwen3_cline_roocode:4b",
    debug=False,
    rolling="expanding",
    train_window=5,
)
```

Depois rode:

```bash
conda activate agno
cd /home/anp/Documents/lucas_mestrado/Statistics_and_Seq2Seq
python run_tsf_orchestrator.py
```

### 8.3 Rodar apenas um dataset index (research loop standalone)

O research loop standalone está em `orchestrator/run_research_loop.py`:

```bash
conda activate agno
cd /home/anp/Documents/lucas_mestrado/Statistics_and_Seq2Seq
python -m orchestrator.run_research_loop --dataset-index 146 --models ARIMA,ETS,THETA --use-llm
```

---

## 9) Saídas e rastreabilidade

Durante a execução:

- O avaliador salva resultados em `context["orchestrator_last_eval"]`.
- O pipeline salva resultado final em `context["orchestrator_last_pipeline"]`.
- No modo 2, também salva candidatos em `context["orchestrator_last_candidates"]`.
- No modo 2, `context["tools_called"]` registra as tools chamadas pelos agentes (ver seção 6).

Isso dá rastreabilidade e evita o modelo “inventar números”: mesmo no modo 2, o ranking/métricas vêm do avaliador determinístico.

### 9.1 CSV (campos explícitos para análise)

Além do blob JSON em `description`, o runner grava colunas explícitas úteis para análise:

- `score_preset`
- `tool_missing`, `tools_called`
- `proposer_selected_names`, `proposer_params_overrides`, `proposer_force_debate`, `proposer_debate_margin`
- `skeptic_remove_names`, `skeptic_add_names`, `skeptic_params_overrides`
- `statistician_remove_names`, `statistician_add_names`, `statistician_params_overrides`
- `final_candidate_names`, `final_candidate_count`
- `best_strategy_name`, `best_strategy_method`, `best_strategy_params`
- `selected_base_models`, `weights_by_horizon`, `predict_debug`
