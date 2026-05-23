# Orchestrator: Arquitetura de Combinação de Modelos de Previsão com Agentes LLM

## Visão Geral

O Orchestrator é um sistema multi-agente para **combinação ótima de previsões de séries temporais**. Dado um conjunto de modelos base (ARIMA, ETS, CatBoost, Random Forest, variantes de wavelets, etc.), o sistema seleciona e combina suas previsões de forma anti-leakage usando agentes LLM colaborativos e avaliação determinística.

**Problema resolvido**: Qual estratégia de combinação (média, ponderação, stacking, seleção por horizonte, etc.) produz a melhor previsão multi-horizonte para uma série específica, dado o comportamento observado nos folds de validação?

**Hipótese central**: Agentes LLM com acesso a estatísticas de validação conseguem propor e refinar um conjunto de estratégias candidatas melhor do que uma seleção puramente automática, especialmente em casos ambíguos onde dados de validação não separam claramente os candidatos.

---

## Arquitetura Geral

```
┌──────────────────────────────────────────────────────────────────────┐
│                        run_tsf_orchestrator.py                       │
│  (loop sobre séries → init_context → run_langchain_pipeline)         │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                   CONTEXT_MEMORY (global shared state)               │
│  all_validations, predictions, models_available, pattern_insights…   │
└──────────────┬────────────────────────────────────┬──────────────────┘
               │                                    │
               ▼                                    ▼
┌──────────────────────────┐        ┌───────────────────────────────┐
│    orchestrator_langchain │        │      orchestrator/             │
│    (LangChain wrappers)   │        │  (núcleo determinístico)       │
│  pipeline.py              │        │  evaluator.py                 │
│  agents.py                │        │  strategies.py                │
│  langchain_tools.py       │        │  final_predictor.py           │
│  context.py               │        │  diagnostics.py               │
│  prompts/                 │        │  tools.py                     │
└──────────────────────────┘        └───────────────────────────────┘
```

### Modos de operação

| Modo | Função | Quando usar |
|------|---------|-------------|
| `run_langchain_pipeline` | LLM + avaliação determinística | `use_llm=True` |
| `run_deterministic_pipeline` | Somente avaliação determinística com candidatos fixos | `use_llm=False` |

---

## Os 4 Agentes LLM

Cada agente é instanciado via `LangchainAgent` (em `orchestrator_langchain/agents.py`), que usa `ChatOllama` apontando para `http://127.0.0.1:11501` (Ollama local). Cada agente tem:
- Um **system prompt** carregado de `orchestrator_langchain/prompts/`
- Uma ou mais **ferramentas LangChain** que acessa dados do contexto
- Um mecanismo de **force_tool_call** que reforça chamada de ferramenta com nudges progressivos se o LLM tentar responder sem chamar

### Agente 1: PatternAnalyst

| Atributo | Valor |
|----------|-------|
| Prompt | `prompts/pattern_analyst.md` |
| Ferramenta | `build_fold_cot_context` |
| Temperatura | 0.3 |
| Modelo padrão no main | `qwen3.5:27b-q4_K_M` |
| Papel | Analista de padrões via STL |

**O que faz**: Analisa os folds de validação usando decomposição STL (Seasonal-Trend-Loess). Calcula correlações de tendência e sazonalidade entre cada modelo e o y_true. Identifica:
- Qual modelo captura melhor a **tendência** (`trend_champion`)
- Qual modelo captura melhor a **sazonalidade** (`seasonality_champion`)
- Diagnósticos avançados: autocorrelação residual (Ljung-Box), heterocedasticidade, drift entre folds, entropia espectral do y_true, tau de Kendall para estabilidade de rankings, similaridade de erros entre modelos (redundância)
- **Flags**: `concept_drift_detected`, `models_redundant`, `ytrue_unpredictable`, `rankings_unstable`

**Output JSON**:
```json
{
  "trend_champion": "ARIMA",
  "seasonality_champion": "ETS",
  "overall_champion": "CWT_catboost",
  "horizon_specialists": {"early": "catboost", "late": "ARIMA"},
  "tier1_models": ["CWT_catboost", "ARIMA", "ETS"],
  "recommended_method_hint": "inverse_rmse_weights",
  "recommended_weighting_basis": "mixed",
  "key_insights": {
    "rmse_spread_ratio": 0.42,
    "high_disagreement": true,
    "high_seasonality_variance": false,
    "ytrue_trend_directions": ["up", "up", "flat"]
  },
  "cot_narrative": "ARIMA lidera em correlação de tendência (0.91). ETS domina sazonalidade (0.87). Spread RMSE alto (0.42) justifica ponderação inversa."
}
```

**Papel no pipeline**: É executado **antes** do Proposer. Seus insights são injetados no brief do Proposer e ficam disponíveis em `CONTEXT_MEMORY["pattern_analyst_insights"]`. Se o agente falhar, o pipeline continua sem seus insights (non-fatal).

---

### Agente 2: Proposer

| Atributo | Valor |
|----------|-------|
| Prompt | `prompts/proposer.md` |
| Ferramenta | `proposer_brief` |
| Temperatura | 0.2 |
| Modelo padrão no main | `gemma4:26b` |
| Papel | Selector inicial de estratégias candidatas |

**O que faz**: Chama `proposer_brief_tool()` para receber o brief completo (estatísticas de validação, biblioteca de candidatos condicionada ao dataset, recomendações de hiperparâmetros, insights do PatternAnalyst). Seleciona um subconjunto de estratégias da biblioteca e escolhe um `score_preset`.

**Output JSON**:
```json
{
  "selected_names": [
    "topk_mean_per_horizon_k5",
    "inverse_rmse_weights_k5_sh0.25",
    "best_per_horizon_by_validation",
    "ridge_stacking_l220_topk3"
  ],
  "params_overrides": {
    "topk_mean_per_horizon_k5": {"top_k": 4},
    "inverse_rmse_weights_k5_sh0.25": {"shrinkage": 0.30}
  },
  "score_preset": "rmse_focus",
  "force_debate": true,
  "debate_margin": 0.03,
  "rationale": "n_unique_winners=4 favorece seleção por horizonte. RMSE spread=0.42 justifica top-k pequeno. Debate forçado por margem estreita entre candidatos."
}
```

**Regras anti-viés** (hardcoded no prompt):
- Proibido propor só `baseline_mean`
- Deve incluir pelo menos 1 candidato de tipo `selection`, `weighted` ou `stacking`
- Deve propor ao menos 3 candidatos
- O `score_preset` vem da recomendação automática do brief (só pode ser override com justificativa explícita)

---

### Agente 3: Skeptic

| Atributo | Valor |
|----------|-------|
| Prompt | `prompts/skeptic.md` |
| Ferramenta | `debate_packet` |
| Temperatura | 0.2 |
| Modelo padrão no main | `gpt-oss:20b` |
| Papel | Auditor de leakage e enforcer de diversidade |

**O que faz**: Recebe o `debate_packet` com ranking dos candidatos atuais, análise de tie-break (Diebold-Mariano + bootstrap pareado) e universo completo de candidatos. Decide quais candidatos remover/adicionar/ajustar.

**Duplo papel**:
1. **Auditor de leakage**: Remove estratégias que usam dados futuros (fitted na janela predita)
2. **Enforcer de diversidade**: Garante que o conjunto tenha candidatos de tipos variados

**Output JSON**:
```json
{
  "add_names": ["dba_combination"],
  "remove_names": ["baseline_mean"],
  "params_overrides": {"inverse_rmse_weights_k5_sh0.25": {"shrinkage": 0.35}},
  "rationale": "relative_spread_mean=0.31 justifica DBA. baseline_mean redundante com topk. Diebold-Mariano p=0.22 (top-1 e top-2 são estatisticamente iguais) → adiciona tipo diverso.",
  "changes": ["added dba_combination for high disagreement", "removed redundant baseline_mean"],
  "when_good": "Quando modelos divergem em amplitude ou fase (spread > 0.3)"
}
```

---

### Agente 4: Statistician

| Atributo | Valor |
|----------|-------|
| Prompt | `prompts/statistician.md` |
| Ferramenta | `debate_packet` |
| Temperatura | 0.2 |
| Modelo padrão no main | `qwen3:14b` |
| Papel | Especialista em robustez e evidência empírica |

**O que faz**: Analisa o mesmo `debate_packet`, focando em evidência estatística para melhorar o conjunto. Prioriza candidatos que funcionam bem dadas as características do dataset (spread de RMSE, janelas de validação, heterogeneidade por horizonte).

**Tabela de decisão** (do prompt):

| Condição | Ação |
|----------|------|
| RMSE spread ratio ≥ 0.3 | Adicionar `inverse_rmse_weights` ou `topk_mean_per_horizon` com k pequeno |
| n_unique_winners ≥ 3 | Adicionar `best_per_horizon_by_validation` |
| relative_spread_mean ≥ 0.25 | Adicionar `dba_combination` ou `trimmed_mean` (trim=0.2) |
| RMSE_std/RMSE > 0.3 | Aumentar shrinkage; usar `robust_median` |
| n_windows ≥ 6 | `ridge_stacking` se torna viável |
| Série sazonal + `models_redundant=true` | Adicionar `stl_hierarchical_stacking` |
| `concept_drift_detected=true` | Preferir `ade_dynamic_error` ou `exp_weighted_average` |

---

## O Protocolo de Debate (Du et al. 2023)

O debate entre Skeptic e Statistician é inspirado em Du et al. (2023) e segue 2 rodadas:

```
                    ┌─────────────────────────────────┐
                    │    Pre-debate: avaliação rápida  │
                    │    (Proposer output → eval_all)  │
                    └──────────────┬──────────────────┘
                                   │
             ┌─────────────────────┴──────────────────────┐
             │           GATILHOS DO DEBATE               │
             │  1. debate=True (forçado globalmente)       │
             │  2. proposer_force_debate=True              │
             │  3. tie estatístico (DM + bootstrap p>0.10) │
             │  4. margem entre top-1 e top-2 < threshold  │
             └─────────────────────┬──────────────────────┘
                                   │
                          debate ativado?
                       ┌────┴────┐
                      Sim       Não
                       │         │
           ┌───────────▼──────────────────────┐
           │  RODADA 1 (independente/cego)     │
           │  Skeptic-R1  ←→  Statistician-R1  │
           │  (cada um responde sem ver o par)  │
           └───────────┬──────────────────────┘
                       │
           ┌───────────▼──────────────────────┐
           │  RODADA 2 (com visibilidade cruzada)│
           │  Skeptic-R2 vê Statistician-R1     │
           │  Statistician-R2 vê Skeptic-R1     │
           │  Cada um aceita/rejeita e revisa   │
           └───────────┬──────────────────────┘
                       │
           ┌───────────▼──────────────────────┐
           │  Aplicação sequencial das ações   │
           │  Skeptic-R2 → Statistician-R2     │
           └───────────┬──────────────────────┘
                       │
           ┌───────────▼──────────────────────┐
           │  Avaliação determinística final   │
           │  (evaluate_all)                   │
           └──────────────────────────────────┘
```

**Tie-break estatístico** (em `orchestrator/diagnostics.py`):
- **Diebold-Mariano (1995)** com correção Harvey-Leybourne-Newbold (1997): testa se a diferença de loss entre top-1 e top-2 é estatisticamente significativa
- **Bootstrap pareado**: reamostra os scores por janela para estimar a distribuição da diferença
- Se ambos os testes não rejeitam a igualdade (p > 0.10), considera-se **empate estatístico** → debate é acionado automaticamente

**Referência**: Du et al. (2023) — Society of Mind protocol; cada agente vê a posição do par e decide se concorda ou diverge, convergindo para um conjunto de candidatos mais robusto.

---

## Ferramentas (Tools) Determinísticas

As ferramentas são funções Python que os agentes LLM chamam. Elas leem/escrevem o `CONTEXT_MEMORY` e retornam JSON estruturado.

### `proposer_brief` → `proposer_brief_tool()`

Retorna ao Proposer:
- **`validation_summary`**: n_windows, horizon, n_models, métricas por modelo (RMSE, SMAPE, MAPE, POCID), best model por horizonte, disagreement score (spread relativo entre modelos)
- **`candidate_library`**: universo completo de estratégias condicionado ao dataset (ver seção abaixo)
- **`recommended_knobs`**: sugestão determinística de top_k, shrinkage, l2, trim_ratio baseada em n_windows e n_models
- **`score_preset_recommendation`**: preset recomendado automaticamente com base nas métricas
- **`pattern_analyst_insights`**: insights do PatternAnalyst (se disponível)

**Lógica de recomendação de preset** (automática, não-LLM):
```python
if avg_pocid < 45:          → "direction_focus"  # POCID baixo → focar direção
elif rmse_spread > 0.5:     → "rmse_focus"        # modelos muito dispersos
elif avg_smape > 0.3:       → "robust_smape"      # série com escala problemática
else:                       → "balanced"           # default
```

### `debate_packet` → `build_debate_packet_tool()`

Retorna ao Skeptic e Statistician:
- Ranking dos candidatos atuais com métricas
- Análise de tie-break (Diebold-Mariano + bootstrap)
- Universo completo de estratégias disponíveis
- Recomendações de knobs

### `build_fold_cot_context` → `build_fold_cot_context_tool()`

Retorna ao PatternAnalyst:
- Decomposição STL de y_true e de cada modelo por fold
- Métricas avançadas por modelo (bias por horizonte, Ljung-Box, heterocedasticidade, drift)
- Diagnósticos de série (entropia espectral, Hurst, Kendall tau, similaridade de erros)
- Rankings e flags (`concept_drift_detected`, `models_redundant`, etc.)

---

## Biblioteca de Estratégias Candidatas

A biblioteca é gerada **deterministicamente** condicionada ao dataset em `_candidate_universe_from_summary()` e `_suggest_candidates_from_summary()`. Não é fixa — os nomes dos candidatos incluem os hiperparâmetros, e são condicionados a n_models, n_windows, disagreement score, etc.

### Tipos de estratégia

| Tipo | Estratégias disponíveis | Aprende pesos? |
|------|------------------------|----------------|
| `baseline` | `baseline_mean`, `robust_median`, `trimmed_mean_r{tr}`, `dba_combination` | Não |
| `selection` | `best_single_by_validation`, `best_per_horizon_by_validation`, `topk_mean_per_horizon_k{k}` | Não |
| `weighted` | `inverse_rmse_weights_k{k}_sh{sh}`, `exp_weighted_average_eta{η}_trim{tr}`, `poly_weighted_average_p{p}_trim{tr}`, `ade_dynamic_error_beta{β}_trim{tr}` | Sim (anti-leakage) |
| `stacking` | `ridge_stacking_l2{l2}_topk{k}`, `stl_hierarchical_stacking_p{p}_sh{sh}` | Sim (anti-leakage) |

### Referências bibliográficas das estratégias

- **DBA** (DTW Barycenter Averaging): Petitjean et al. (2011)
- **EWA/PWA** (Exponential/Polynomial Weighted Averaging): Cesa-Bianchi & Lugosi (2006), Gaillard & Goude (2015)
- **ADE** (Arbitrage of forecasting experts, dynamic): Cerqueira et al. (2019)
- **Ridge stacking**: Stock & Watson (2004), Timmermann (2006)
- **STL hierarchical stacking**: Cleveland et al. (1990), Stock & Watson (2004), Timmermann (2006)
- **Diebold-Mariano test**: Diebold & Mariano (1995), Harvey et al. (1997)

### Geração dinâmica de candidatos (condicional ao dataset)

```python
# Exemplos de candidatos gerados para n_models=20, n_windows=3:
top_k = round(sqrt(20)) = 4   # base
shrinkage = 0.35               # poucos windows → mais regularização
l2 = 50.0                      # idem

# Candidatos gerados:
"topk_mean_per_horizon_k4"
"topk_mean_per_horizon_k2"   # conservador
"topk_mean_per_horizon_k6"   # agressivo
"inverse_rmse_weights_k4_sh0.35"
"ridge_stacking_l250_topk3"
"stl_hierarchical_stacking_p6_sh0.0"  # period = horizon//2
"ade_dynamic_error_beta0.5_trim0.8"
# ... etc.
```

Os grids de hiperparâmetros (`k_grid`, `trim_grid`, `shrink_grid`, `l2_grid`) são escalados com n_models e n_windows para evitar explosion combinatorial desnecessária.

---

## Avaliação Determinística (Anti-Leakage)

A função `evaluate_all()` em `orchestrator/evaluator.py` é **completamente determinística** — nenhum LLM participa da decisão final. Para cada candidato:

1. **Gera previsões combinadas** via `generate_combined_predictions()` usando rolling/expanding windows
2. **Calcula métricas** por horizonte: MAPE, SMAPE, RMSE, POCID
3. **Normaliza** todas as métricas em relação ao `baseline_mean` (razão candidate/baseline)
4. **Computa score composto**: `score = a·RMSE_n + b·SMAPE_n + c·MAPE_n - d·POCID_n`

### Score Presets

| Preset | a_RMSE | b_SMAPE | c_MAPE | d_POCID | Quando usar |
|--------|--------|---------|--------|---------|-------------|
| `balanced` | 0.30 | 0.30 | 0.20 | 0.20 | Default |
| `rmse_focus` | 0.50 | 0.20 | 0.20 | 0.10 | Alta dispersão entre modelos |
| `direction_focus` | 0.25 | 0.25 | 0.10 | 0.40 | Baixo POCID (< 45) |
| `robust_smape` | 0.20 | 0.50 | 0.10 | 0.20 | SMAPE médio > 0.3 |

**Score menor = melhor** (exceto POCID que entra com sinal negativo).

### Anti-leakage garantido

Todas as estratégias que aprendem pesos (weighted, stacking) usam **apenas janelas passadas** para estimar os pesos e aplicam nas janelas futuras — nunca a janela sendo predita está no conjunto de treinamento dos pesos.

---

## Pipeline Completo: Passo a Passo

```
Entrada: dataset_index, modelos base treinados, folds de validação
```

### Passo 0 — Inicialização do contexto

```python
init_context()
CONTEXT_MEMORY["models_available"] = models
generate_all_validations_context(models, i, train_window=3, dataset=dataset)
```

Lê os CSVs de previsão de cada modelo base e monta:
- `all_validations["predictions"]`: lista de dicts `{model: [preds]}` por janela de validação
- `all_validations["test"]`: valores reais correspondentes
- `predictions`: previsões de cada modelo no período de teste final (horizonte alvo)

### Passo 1 — PatternAnalyst (não-fatal)

O PatternAnalyst chama `build_fold_cot_context` que executa:
- Decomposição STL em cada fold para cada modelo e para y_true
- Diagnósticos: Ljung-Box, heterocedasticidade, drift, entropia espectral, Kendall tau, similaridade de erros
- Retorna JSON com champions, tier lists, method hint e flags

Resultado armazenado em `CONTEXT_MEMORY["pattern_analyst_insights"]`.

### Passo 2 — Proposer

O Proposer chama `proposer_brief` que retorna o brief completo. O LLM seleciona candidatos da biblioteca e define o score_preset.

Validações server-side:
- Nomes desconhecidos são resolvidos via `resolve_unknown_candidate()` (regex pattern matching sobre variantes de nomes)
- `params_overrides` com chaves inválidas são silenciosamente descartadas (warning)
- Candidatos com menos de 2 no conjunto final recebem `baseline_mean` como fallback

### Passo 3 — Avaliação pré-debate

Avaliação determinística do conjunto proposto pelo Proposer. Detecta:
- Margem de score entre top-1 e top-2
- Empate estatístico via Diebold-Mariano + bootstrap

### Passo 4 — Debate (condicional)

**Gatilhos**:
1. `debate=True` (parâmetro global)
2. `force_debate=True` (Proposer decidiu)
3. Empate estatístico (DM p > 0.10 AND bootstrap p > 0.10)
4. Margem top-1/top-2 < threshold (default: 0.02)

Se debate é ativado:
- **Rodada 1**: Skeptic e Statistician respondem independentemente ao `debate_packet`
- **Rodada 2**: Cada agente vê o JSON compacto do par (add/remove/params_overrides/rationale) e revisa sua posição
- Ações R2 são aplicadas sequencialmente: Skeptic primeiro, depois Statistician

### Passo 5 — Avaliação determinística final

`evaluate_all()` avalia todos os candidatos pós-debate, ranqueia por score composto e seleciona o melhor.

### Passo 6 — Previsão final

`predict_final_from_context()` aplica a estratégia vencedora sobre as previsões do período de teste final (usando pesos aprendidos nos folds de validação).

### Passo 7 — Persistência

Métricas (MAPE, SMAPE, RMSE, MAE, POCID), previsões, trace de decisão (debate trace, agente logs, think blocks) e artifacts LLM são escritos no CSV de resultados e no JSON de artifacts.

---

## Proteções contra Alucinação LLM

### Validação de nomes (`_validate_actions_against_universe`)

Antes de aplicar qualquer ação LLM (add/remove/override), o pipeline valida:
1. `add_names` → cada nome deve estar no universo; se não estiver, tenta `resolve_unknown_candidate()`
2. `remove_names` → deve estar no conjunto atual
3. `params_overrides` → chaves inválidas emitem warning e são descartadas (não hard-stop)
4. Promoção automática: se `params_overrides` referencia um nome válido que não estava em `add_names`, ele é promovido silenciosamente

### Resolver de nomes (`resolve_unknown_candidate`)

Quando o LLM inventa um nome como `"inv_rmse_k3_shrink0.25"`, o resolver usa regex para mapeá-lo para o canônico `"inverse_rmse_weights_k3_sh0.25"`. Suporta variações para todos os tipos de estratégia.

### Retry com retry logic (`_run_agent_with_retry`)

Cada agente tem até 3 tentativas para retornar JSON válido. Se o LLM responde com texto ao invés de chamar a ferramenta, um nudge progressivo é enviado:
- Tentativa 1: "ERROR: You did NOT call any tool..."
- Tentativa 2: "CRITICAL: This is your LAST chance. Call the tool NOW..."

### Clamping de hiperparâmetros

```python
top_k      → clamp(2, n_models)
trim_ratio → clamp(0.0, 0.4)
shrinkage  → clamp(0.0, 0.9)
l2         → clamp(0.1, 1000.0)
period     → clamp(2, 24)
```

### Mínimo de 2 candidatos

Se ações LLM removeram demais, `baseline_mean` é reinserido como fallback.

---

## Exemplo Completo: Entrada → Saída

### Entrada (contexto para série index=7, dataset ETTH1)

```
n_models=20, n_windows=3, horizon=24
models: ARIMA, ETS, THETA, rf, catboost, CWT_rf, DWT_rf, FT_rf, CWT_catboost, DWT_catboost, 
        FT_catboost, ONLY_CWT_catboost, ONLY_CWT_rf, ONLY_DWT_catboost, ONLY_DWT_rf, 
        ONLY_FT_catboost, ONLY_FT_rf, NaiveSeasonal, NaiveMovingAverage, ONLY_FT_rf

Dados de validação (3 janelas, horizonte 24):
 window 0 → all_validations["predictions"][0]["ARIMA"] = [2.1, 2.0, 1.9, ...]
 window 0 → all_validations["test"][0] = [2.0, 1.95, 1.85, ...]
 ...
```

### Passo 1 — PatternAnalyst output

```json
{
  "trend_champion": "ARIMA",
  "seasonality_champion": "ETS",
  "overall_champion": "CWT_catboost",
  "recommended_method_hint": "inverse_rmse_weights",
  "key_insights": {
    "rmse_spread_ratio": 0.42,
    "high_disagreement": false
  },
  "cot_narrative": "ARIMA lidera em trend_corr=0.91. ETS em seas_corr=0.87. CWT_catboost tem menor RMSE agregado. Spread 0.42 sugere ponderação por performance."
}
```

### Passo 2 — Proposer output

```json
{
  "selected_names": [
    "inverse_rmse_weights_k4_sh0.35",
    "topk_mean_per_horizon_k4",
    "best_per_horizon_by_validation",
    "ridge_stacking_l250_topk3"
  ],
  "params_overrides": {
    "inverse_rmse_weights_k4_sh0.35": {"shrinkage": 0.35, "top_k": 4}
  },
  "score_preset": "rmse_focus",
  "force_debate": false,
  "debate_margin": 0.02,
  "rationale": "RMSE spread=0.42 favorece ponderação inversa. 3 unique winners por horizonte justifica best_per_horizon. n_windows=3 → shrinkage=0.35."
}
```

### Passo 3 — Avaliação pré-debate

```
Ranking pré-debate:
1. inverse_rmse_weights_k4_sh0.35  score=0.721
2. topk_mean_per_horizon_k4         score=0.724
3. best_per_horizon_by_validation   score=0.731
4. ridge_stacking_l250_topk3        score=0.748

Margem top-2: 0.724 - 0.721 = 0.003 < 0.02 → DEBATE ACIONADO
DM test p=0.31, bootstrap p=0.28 → empate estatístico confirmado
```

### Passo 4 — Debate (2 rodadas)

**Skeptic-R1**:
```json
{
  "add_names": ["dba_combination"],
  "remove_names": [],
  "params_overrides": {},
  "rationale": "top-1 e top-2 estatisticamente iguais (DM p=0.31). Adiciona tipo diferente para diversidade."
}
```

**Statistician-R1**:
```json
{
  "add_names": ["stl_hierarchical_stacking_p12_sh0.0"],
  "remove_names": ["ridge_stacking_l250_topk3"],
  "params_overrides": {"topk_mean_per_horizon_k4": {"top_k": 3}},
  "rationale": "n_windows=3 torna ridge instável. STL decomp-aware é mais robusto. top_k=3 < sqrt(20)=4 para regularizar."
}
```

**Skeptic-R2** (leu Statistician-R1):
```json
{
  "add_names": ["dba_combination"],
  "remove_names": ["ridge_stacking_l250_topk3"],
  "params_overrides": {},
  "rationale": "Concordo com remoção do ridge (n_windows=3 insuficiente). Mantendo dba_combination para diversidade. STL proposto pelo Statistician cobre componentes."
}
```

**Statistician-R2** (leu Skeptic-R1):
```json
{
  "add_names": ["stl_hierarchical_stacking_p12_sh0.0"],
  "remove_names": ["ridge_stacking_l250_topk3"],
  "params_overrides": {"topk_mean_per_horizon_k4": {"top_k": 3}},
  "rationale": "Concordo com dba_combination do Skeptic. Mantendo STL e remoção do ridge."
}
```

**Conjunto final pós-debate**:
```
inverse_rmse_weights_k4_sh0.35
topk_mean_per_horizon_k4 (top_k→3)
best_per_horizon_by_validation
dba_combination
stl_hierarchical_stacking_p12_sh0.0
```

### Passo 5 — Avaliação final

```
Ranking final:
1. inverse_rmse_weights_k4_sh0.35   score=0.712  ← VENCEDOR
2. stl_hierarchical_stacking_p12    score=0.718
3. topk_mean_per_horizon_k3         score=0.721
4. dba_combination                  score=0.735
5. best_per_horizon_by_validation   score=0.739
```

### Passo 6 — Previsão final

Aplica `inverse_rmse_weights_k4_sh0.35` sobre as previsões dos modelos base no período de teste final:
```
Para cada horizonte h=1..24:
  1. Seleciona top-4 modelos por RMSE nos folds de validação
  2. Calcula pesos: w_m ∝ 1/(RMSE_m + ε)
  3. Aplica shrinkage=0.35: w = 0.65 * w_inv_rmse + 0.35 * uniform
  4. Projeta no simplex (w ≥ 0, sum(w) = 1)
  5. y_hat(h) = Σ_m w_m(h) * pred_m(h)
```

### Output final registrado no CSV

```
dataset_index=7, horizon=24, regressor=orchestrator_llm_v1_pattern
mape=0.0432, smape=0.0419, rmse=0.0891, mae=0.0673, pocid=62.5
best_strategy_name=inverse_rmse_weights_k4_sh0.35
best_strategy_method=inverse_rmse_weights_per_horizon
debate_ran=True, debate_trigger=auto_statistical_tie
approach_pre_debate=inverse_rmse_weights_k4_sh0.35
approach_post_debate=inverse_rmse_weights_k4_sh0.35
proposer_selected_names=["inverse_rmse_weights_k4_sh0.35", ...]
```

---

## Estrutura de Arquivos

```
Statistics_and_Seq2Seq/
├── run_tsf_orchestrator.py          # Entrypoint: loop sobre séries/datasets
│
├── orchestrator_langchain/          # Camada LangChain (wrappers LLM)
│   ├── pipeline.py                  # run_langchain_pipeline (monkey-patches base)
│   ├── agents.py                    # LangchainAgent, factory functions
│   ├── langchain_tools.py           # Ferramentas LangChain (@tool decorators)
│   ├── context.py                   # CONTEXT_MEMORY + helpers + read_model_preds
│   └── prompts/
│       ├── pattern_analyst.md       # System prompt do PatternAnalyst
│       ├── proposer.md              # System prompt do Proposer
│       ├── skeptic.md               # System prompt do Skeptic
│       ├── statistician.md          # System prompt do Statistician
│       └── orchestrator.md          # (legado, não usado no pipeline atual)
│
└── orchestrator/                    # Núcleo determinístico
    ├── pipeline.py                  # run_llm_pipeline, run_deterministic_pipeline
    ├── tools.py                     # proposer_brief_tool, debate_packet_tool, build_fold_cot_context_tool
    ├── evaluator.py                 # evaluate_all, evaluate_candidate, ScoreConfig
    ├── strategies.py                # generate_combined_predictions, implementações de combinação
    ├── final_predictor.py           # predict_final_from_context
    ├── diagnostics.py               # DM test, bootstrap, Ljung-Box, spectral entropy, Hurst, Kendall tau
    ├── schemas.py                   # CandidateStrategy, parse_candidates
    ├── data_contract.py             # ValidationData, load_validation_from_context
    ├── metrics.py                   # rmse_safe, smape_safe, mape_safe, pocid_within_sequence
    ├── utils.py                     # extract_json_object, strip_think_blocks, ModelConfig
    └── agents.py                    # Factory functions base (para modo sem LangChain)
```

---

## Modelos LLM Configurados no Main

```python
# run_tsf_orchestrator.py __main__
proposer_model       = ModelConfig(model="gemma4:26b",      temperature=0.7)
skeptic_model        = ModelConfig(model="gpt-oss:20b",     temperature=0.3)
statistician_model   = ModelConfig(model="qwen3:14b",       temperature=0.2)
pattern_analyst_model= ModelConfig(model="qwen3.5:27b-q4_K_M", temperature=0.2)
```

Todos são servidos localmente via Ollama em `http://127.0.0.1:11501`.

---

## Outputs e Rastreabilidade

### Arquivos gerados por execução

```
timeseries/mestrado/resultados/orchestrator_llm_v1_pattern/
├── {DATASET}.csv                       # Métricas + trace de decisão (1 linha por série)
└── llm_artifacts/{DATASET}/
    └── dataset_{i}.json                # Prompts, outputs raw, JSON parseados por agente
```

### Colunas rastreadas no CSV

- **Métricas**: `mape`, `smape`, `rmse`, `msmape`, `mae`, `pocid`
- **Estratégia**: `best_strategy_name`, `best_strategy_method`, `best_strategy_params`
- **Trace de debate**: `debate_ran`, `debate_trigger`, `approach_pre_debate`, `approach_post_debate`
- **Ações dos agentes**: `proposer_selected_names`, `skeptic_remove_names`, `skeptic_add_names`, `statistician_remove_names`, `statistician_add_names`
- **Think blocks** (se o modelo suportar): `proposer_think`, `skeptic_think`, `statistician_think`, `pattern_analyst_think`
- **PatternAnalyst**: `pattern_analyst_trend_champion`, `pattern_analyst_seas_champion`, `pattern_analyst_method_hint`, `pattern_analyst_narrative`
- **Pesos**: `weights_by_horizon`, `selected_base_models`

---

## Referências

- **Du et al. (2023)** — Improving Factuality and Reasoning in Language Models through Multiagent Debate
- **Diebold & Mariano (1995)** — Comparing Predictive Accuracy
- **Harvey, Leybourne & Newbold (1997)** — Testing the equality of prediction mean squared errors
- **Petitjean et al. (2011)** — A global averaging method for dynamic time warping (DBA)
- **Cesa-Bianchi & Lugosi (2006)** — Prediction, Learning, and Games (EWA/PWA)
- **Gaillard & Goude (2015)** — Forecasting Electricity Consumption by Aggregating Experts
- **Cerqueira et al. (2019)** — Arbitrage of Forecasting Experts (ADE)
- **Cleveland et al. (1990)** — STL: A Seasonal-Trend Decomposition Procedure Based on Loess
- **Stock & Watson (2004)** — Combination Forecasts of Output Growth in a Seven-Country Data Set
- **Timmermann (2006)** — Forecast Combinations (Handbook of Economic Forecasting)
