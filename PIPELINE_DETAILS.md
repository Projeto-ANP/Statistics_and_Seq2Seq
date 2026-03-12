# Detalhamento Completo do Pipeline Multi-Agentes para Forecasting

> **Objetivo deste documento**: Explicar, passo a passo e com detalhe técnico total, como funciona o pipeline que combina previsões de múltiplos modelos de séries temporais. Cada agente, cada métrica, cada equação e cada condição de ramificação estão aqui descritos como numa história — do dado bruto até a previsão final.

---

## Índice

1. [De onde vêm os dados](#1-de-onde-vêm-os-dados)
2. [Estrutura de Janelas: o Coração Anti-Leakage](#2-estrutura-de-janelas-o-coração-anti-leakage)
3. [Visão Geral do Pipeline de Execução](#3-visão-geral-do-pipeline-de-execução)
4. [Agente 1 — PatternAnalyst](#4-agente-1--patternanalyst)
5. [Agente 2 — Proposer](#5-agente-2--proposer)
6. [O Gatilho do Debate](#6-o-gatilho-do-debate)
7. [Agente 3 — Skeptic/Auditor](#7-agente-3--skepticauditor)
8. [Agente 4 — Statistician](#8-agente-4--statistician)
9. [O Orquestrador Determinístico](#9-o-orquestrador-determinístico)
10. [As Métricas de Avaliação — Por Que Cada Uma?](#10-as-métricas-de-avaliação--por-que-cada-uma)
11. [Composite Score — A Fórmula do Ranking Final](#11-composite-score--a-fórmula-do-ranking-final)
12. [Estratégias de Combinação e Pesos (Matemática Completa)](#12-estratégias-de-combinação-e-pesos-matemática-completa)
13. [Proteção Matemática: Projeção no Simplex](#13-proteção-matemática-projeção-no-simplex)
14. [Previsão Final: da Melhor Estratégia ao Número](#14-previsão-final-da-melhor-estratégia-ao-número)


---

## 1. De onde vêm os dados

Imagine que você tem vários modelos de previsão — ARIMA, ETS, SVR, uma rede neural — e cada um deles fez suas predições para uma determinada série temporal. Esses resultados estão salvos num arquivo chamado `results.csv`.

Esse arquivo contém, por linha, as previsões de cada modelo para um "horizonte" (múltiplos passos de tempo futuros). Há previsões de **múltiplas janelas históricas** chamadas `val1, val2, val3, ...` e uma previsão chamada `final`, que é o ponto **cego**, ou seja, o único período em que existe um valor real escondido que servirá de teste verdadeiro do sistema.

**Regra de Ouro**: Nenhum agente, nenhuma função, e nenhum modelo da combinação jamais vê diretamente os valores reais da janela `final` durante a tomada de decisão sobre estratégias.

---

## 2. Estrutura de Janelas: o Coração Anti-Leakage

O sistema emprega a técnica de **validação cruzada com origem expandindo ou rolando** (*expanding/rolling origin cross-validation*). Pense assim:

```
Linha do tempo de janelas:

[val1] → [val2] → [val3] → [val4]  |  [final]  (CEGO/TESTE)
```

- Ao avaliar o desempenho de uma estratégia na **janela `i`**, o sistema usa para treinar/aprender **somente as janelas `0` até `i-1`**.
- Ao simular como teria sido a decisão no momento `val3`, o sistema enxerga apenas `val1` e `val2`. **`val3` em si é o alvo.** `val4` e `final` são invisíveis.  
- Esse fatiamento é implementado pela função `_train_slice(i, cfg)`:

```python
# Modo EXPANDING (janela cresce com o tempo):
slice(0, i)  # janelas de 0 até i-1

# Modo ROLLING (janela de tamanho fixo):
slice(max(0, i - train_window), i)
```

Um guard rígido garante que mesmo que um agente LLM alucinado passe `train_window=999` para um dataset com apenas 3 janelas, a fatia sempre começa em `max(0, ...)`, impedindo índices negativos.

---

## 3. Visão Geral do Pipeline de Execução

O pipeline completo segue esta sequência orquestrada no `pipeline.py`:

```
DADOS (results.csv)
     │
     ▼
[Step 0] PatternAnalyst  ──► Identifica champions de Trend e Seasonality
     │
     ▼
[Step 1] Proposer        ──► Propõe 3–12 estratégias candidatas de combinação
     │
     ▼
[Step 2] Avaliação Pré-Debate ──► Determina se os Top-2 estão muito próximos
     │
     ├──► [SE EMPATE ACIRRADO ou FORÇADO]
     │          │
     │          ▼
     │    [Step 3] Skeptic  ──► Remove candidatos com leakage ou inválidos
     │          │
     │          ▼
     │    [Step 4] Statistician ──► Adiciona robustez quantitativa
     │
     ▼
[Step 5] Orquestrador Determinístico ──► Avalia TODAS candidatas no passado
     │
     ▼
[Step 6] Melhor Estratégia + Previsão Final (usando a janela final)
```

---

## 4. Agente 1 — PatternAnalyst

### O que ele é

O `PatternAnalyst` é o especialista em decomposição de séries temporais. É o **primeiro agente a falar** no pipeline. Ele não propõe nada — seu único trabalho é entender matematicamente o comportamento passado de cada modelo e da série real.

Temperatura: `0.3` (levemente criativo para explorar padrões).

### O que ele faz (Passo a Passo)

**1. Chama a tool `build_fold_cot_context`**

Essa função Python (não o LLM) executa o trabalho pesado. Ela pega todas as janelas de validação históricas e, para cada modelo e para a série real (`y_true`), calcula:

- **Decomposição STL (Seasonal-Trend decomposition using LOESS)**: Usa o algoritmo STL do `statsmodels.tsa.seasonal.STL` para separar a série em 3 componentes: **tendência**, **sazonalidade** e **resíduo**. É mais robusto que a decomposição linear simples.
- **Correlação de Tendência**: `Pearson(trend_modelo, trend_y_true)` — correlação entre a componente de tendência do modelo e a tendência real.
- **Correlação Sazonal**: `Pearson(seasonal_modelo, seasonal_y_true)` — correlação entre a componente sazonal do modelo e a sazonalidade real.
- Rankings de **Early Horizon** (1º terço do horizonte) e **Late Horizon** (último terço) por RMSE (métricas aprovadas: RMSE, SMAPE, MAPE, POCID).
- Flags como `high_model_disagreement` (quando `rmse_spread_ratio > 0.3`) e `high_seasonality_variance` (quando `seas_corr_variance > 0.3`).
- **Valores brutos de STL para decisão da LLM**: A tool retorna os valores de tendência e sazonalidade por fold para que a LLM analise e **decida** quais modelos são os champions (em vez de usar rankings pré-computados).

**2. Recebe o resultado da tool — Exemplo de Output da Tool**

```json
{
  "n_folds_analyzed": 3,
  "horizon": 12,
  "n_models": 4,
  "decomposition_method": "STL (Seasonal-Trend decomposition using LOESS)",
  "ytrue_stl_decomposition": {
    "per_fold": [
      {
        "fold": 0,
        "trend_values": [100.1, 100.5, 101.2, ...],
        "seasonal_values": [-0.5, 0.8, 1.2, ...],
        "trend_direction": "up",
        "trend_strength": 2.31,
        "seasonal_amplitude": 1.45
      },
      {"fold": 1, ...},
      {"fold": 2, ...}
    ],
    "concatenated_trend": [...],
    "concatenated_seasonality": [...],
    "overall_seasonal_amplitude": 1.67
  },
  "model_stl_decomposition": {
    "ARIMA": {
      "trend_per_fold": [[...], [...], [...]],
      "seasonal_per_fold": [[...], [...], [...]],
      "avg_trend_corr": 0.92,
      "avg_seasonal_corr": 0.78
    },
    "ETS": {
      "avg_trend_corr": 0.85,
      "avg_seasonal_corr": 0.94
    },
    "SVR": {
      "avg_trend_corr": 0.61,
      "avg_seasonal_corr": 0.42
    }
  },
  "model_metrics": {
    "ARIMA": {
      "avg_trend_corr": 0.92,
      "avg_seasonal_corr": 0.78,
      "avg_rmse": 4.2,
      "avg_smape": 0.08,
      "early_horizon_rmse": 3.1,
      "late_horizon_rmse": 5.4
    },
    "ETS": {
      "avg_trend_corr": 0.85,
      "avg_seasonal_corr": 0.94,
      "avg_rmse": 4.8,
      "avg_smape": 0.09,
      "early_horizon_rmse": 4.2,
      "late_horizon_rmse": 5.1
    }
  },
  "rmse_rankings": {
    "overall": [{"model": "ARIMA", "avg_rmse": 4.2}, {"model": "ETS", "avg_rmse": 4.8}],
    "early_horizon_specialists": [{"model": "ARIMA", "early_horizon_rmse": 3.1}],
    "late_horizon_specialists":  [{"model": "ETS",   "late_horizon_rmse": 5.1}],
    "worst_by_rmse": [{"model": "SVR", "avg_rmse": 7.8}]
  },
  "model_tiers": {
    "tier1_best": ["ARIMA", "ETS"],
    "tier2_mid":  ["SVR"],
    "tier3_worst": []
  },
  "insights": {
    "rmse_spread_ratio": 0.42,
    "seasonal_corr_variance": 0.18,
    "high_model_disagreement": true,
    "high_seasonality_variance": false
  },
  "llm_decision_required": {
    "trend_champion": "Analyze model_stl_decomposition.avg_trend_corr to decide which model best captures y_true trend",
    "seasonality_champion": "Analyze model_stl_decomposition.avg_seasonal_corr to decide which model best captures y_true seasonality",
    "recommended_method": "Based on insights and model performance, decide the best combination method"
  }
}
```

**Observação Importante**: Note que a tool **não pré-computa** os champions. Ela retorna os valores brutos de correlação (`avg_trend_corr`, `avg_seasonal_corr`) e o campo `llm_decision_required` instrui a LLM a **analisar esses valores e decidir** quais modelos são os melhores captadores de tendência e sazonalidade.

**3. Raciocina (bloco `<think>...</think>`)**

O LLM usa essas informações para responder perguntas como:
- A série real está subindo, descendo ou plana?
- Qual modelo acompanha melhor a inclinação da reta real?
- Qual modelo captura melhor os padrões cíclicos residuais?
- Os modelos concordam entre si? Ou há grande dispersão de RMSE?

**4. Retorna JSON estruturado — Output Final do `PatternAnalyst`**

**Importante**: A LLM agora **analisa os dados brutos de STL** retornados pela tool e **decide** quem são os champions, em vez de simplesmente copiar rankings pré-computados.

```json
{
  "trend_champion": "ARIMA",
  "trend_champion_reasoning": "ARIMA tem avg_trend_corr=0.92 (maior correlação com a tendência real)",
  "seasonality_champion": "ETS",
  "seasonality_champion_reasoning": "ETS tem avg_seasonal_corr=0.94 (maior correlação com a sazonalidade real)",
  "overall_champion": "ARIMA",
  "horizon_specialists": {
    "early": "ARIMA",
    "early_reasoning": "ARIMA tem early_horizon_rmse=3.1 (menor RMSE no 1º terço)",
    "late": "ETS",
    "late_reasoning": "ETS tem late_horizon_rmse=5.1 (menor RMSE no último terço)"
  },
  "tier1_models": ["ARIMA", "ETS"],
  "tier2_models": ["SVR"],
  "recommended_method_hint": "inverse_rmse_weights",
  "recommended_weighting_basis": "rmse",
  "key_insights": {
    "decomposition_method": "STL (LOESS)",
    "rmse_spread_ratio": 0.42,
    "high_disagreement": true,
    "high_seasonality_variance": false,
    "ytrue_trend_direction": "up",
    "ytrue_seasonal_amplitude": 1.67
  },
  "cot_narrative": "Analizei model_stl_decomposition: ARIMA tem avg_trend_corr=0.92 (melhor alinhamento com tendência real). ETS tem avg_seasonal_corr=0.94 (melhor captura de sazonalidade). O alto RMSE spread (0.42) justifica pesos inversamente proporcionais ao erro."
}
```

### Como é identificado o Trend Champion (Decisão da LLM)

A tool calcula a **correlação entre a componente de tendência do modelo e a tendência real** usando STL:

$$\text{avg\_trend\_corr}_m = \frac{1}{F} \sum_{f=1}^{F} \text{Pearson}\left(\text{trend}_{m,f},\ \text{trend}_{\text{real},f}\right)$$

onde:
- $\text{trend}_{m,f}$ = componente de tendência do modelo $m$ no fold $f$, extraída via STL
- $\text{trend}_{\text{real},f}$ = componente de tendência do y_true no fold $f$
- $F$ = número de folds de validação

**A LLM analisa todos os valores de `avg_trend_corr`** no campo `model_stl_decomposition` e **decide** qual modelo é o **Trend Champion** (maior correlação = melhor alinhamento com a tendência real).

### Como é identificado o Seasonality Champion (Decisão da LLM)

A tool calcula a **correlação entre a componente sazonal do modelo e a sazonalidade real** usando STL:

$$\text{avg\_seasonal\_corr}_m = \frac{1}{F} \sum_{f=1}^{F} \text{Pearson}\left(\text{seasonal}_{m,f},\ \text{seasonal}_{\text{real},f}\right)$$

onde:
- $\text{seasonal}_{m,f}$ = componente sazonal do modelo $m$ no fold $f$, extraída via STL
- $\text{seasonal}_{\text{real},f}$ = componente sazonal do y_true no fold $f$

**A LLM analisa todos os valores de `avg_seasonal_corr`** e **decide** qual modelo é o **Seasonality Champion** (maior correlação = melhor captura da sazonalidade real).

### Por que a LLM decide em vez de usar rankings pré-computados?

1. **Contexto completo**: A LLM pode considerar múltiplos fatores (correlação + RMSE + insights) em uma decisão holistica
2. **Raciocínio explícito**: A LLM explica sua escolha no campo `cot_narrative`
3. **Flexibilidade**: Em casos de empate ou valores muito próximos, a LLM pode usar critérios secundários

---

## 5. Agente 2 — Proposer

### O que ele é

O `Proposer` é o **estrategista**. Ele lê o que o `PatternAnalyst` descobriu e transforma isso em uma lista de estratégias de combinação concretas para serem testadas.

Temperatura: `0.2` (mais objetivo, menos criativo).

### O que ele faz (Passo a Passo)

**1. Chama a tool `proposer_brief`**

Essa ferramenta Python prepara um "briefing completo", que contém:

```json
{
  "validation_summary": {
    "n_windows": 3,
    "n_models": 4,
    "models": {
      "ARIMA": {"avg_rmse": 4.2, "std_rmse": 0.8},
      "ETS":   {"avg_rmse": 5.1, "std_rmse": 1.2},
      "SVR":   {"avg_rmse": 7.8, "std_rmse": 2.1},
      "LSTM":  {"avg_rmse": 6.3, "std_rmse": 1.9}
    },
    "best_per_horizon": {
      "n_unique_winners": 3,
      "winners_by_horizon": ["ARIMA", "ETS", "ARIMA", "ARIMA", "ETS", "ETS"]
    },
    "disagreement": {
      "relative_spread_mean": 0.31
    }
  },
  "candidate_library": {
    "candidates": [
      {"name": "baseline_mean", "type": "baseline", "params": {"method": "mean"}},
      {"name": "baseline_median", "type": "baseline", "params": {"method": "median"}},
      {"name": "trimmed_mean_20", "type": "baseline", "params": {"method": "trimmed_mean", "trim_ratio": 0.2}},
      {"name": "best_single_rolling", "type": "selection", "params": {"method": "best_single"}},
      {"name": "best_per_horizon_rolling", "type": "selection", "params": {"method": "best_per_horizon"}},
      {"name": "topk_mean_per_horizon_k3", "type": "selection", "params": {"method": "topk_mean_per_horizon", "top_k": 3}},
      {"name": "inv_rmse_weights_per_horizon_k3_shrink02", "type": "weighted", "params": {"method": "inverse_rmse_weights_per_horizon", "top_k": 3, "shrinkage": 0.2}},
      {"name": "ridge_stacking_per_horizon_l2_10", "type": "stacking", "params": {"method": "ridge_stacking_per_horizon", "l2": 10.0}},
      {"name": "dba_combination", "type": "weighted", "params": {"method": "dba"}},
      {"name": "exp_weighted_per_horizon_eta1", "type": "weighted", "params": {"method": "exp_weighted_average_per_horizon", "eta": 1.0}}
    ]
  },
  "recommended_knobs": {
    "top_k": 3,
    "shrinkage": 0.2,
    "trim_ratio": 0.2
  },
  "pattern_analyst_insights": {
    "trend_champion": "ARIMA",
    "trend_champion_reasoning": "avg_trend_corr=0.92 (maior correlação STL)",
    "seasonality_champion": "ETS",
    "seasonality_champion_reasoning": "avg_seasonal_corr=0.94 (maior correlação STL)",
    "model_stl_decomposition": {"ARIMA": {"avg_trend_corr": 0.92, "avg_seasonal_corr": 0.78}, "ETS": {"avg_trend_corr": 0.85, "avg_seasonal_corr": 0.94}},
    "recommended_method_hint": "inverse_rmse_weights"
  },
  "score_preset_recommendation": {
    "recommended_preset": "rmse_focus"
  }
}
```

**2. Raciocina com base nos sinais**

O Proposer usa uma tabela de decisão interna:

| Sinal do Briefing | Tipo de Estratégia Preferida |
|---|---|
| `n_unique_winners ≥ 3` | `best_per_horizon` ou `topk_mean_per_horizon` |
| `relative_spread_mean ≥ 0.25` | `trimmed_mean` ou `dba` |
| Gap RMSE best/worst ≥ 30% | `inverse_rmse_weights` ou `topk_mean` com k pequeno |
| `n_windows ≤ 4` | `top_k` pequeno (2–3), `shrinkage ≥ 0.3` |
| Trend champion claro | `best_single` ou `inverse_rmse_weights` |

**Regras que ele não pode violar:**
- Deve propor **pelo menos 3 candidatos**.
- Deve incluir pelo menos **1 candidato do tipo `selection`, `weighted` ou `stacking`** (não pode ser tudo `baseline`).
- Só pode referenciar nomes que existem na `candidate_library` — nomes inventados geram hard-stop imediato.

**3. Retorna JSON — Output Final do `Proposer`**

```json
{
  "selected_names": [
    "inv_rmse_weights_per_horizon_k3_shrink02",
    "best_per_horizon_rolling",
    "topk_mean_per_horizon_k3",
    "baseline_median"
  ],
  "params_overrides": {
    "inv_rmse_weights_per_horizon_k3_shrink02": {
      "top_k": 3,
      "shrinkage": 0.25
    },
    "topk_mean_per_horizon_k3": {
      "top_k": 2
    }
  },
  "score_preset": "rmse_focus",
  "force_debate": false,
  "debate_margin": 0.02,
  "rationale": "n_unique_winners=3 justifica per-horizon selection. RMSE spread=0.42 (>30%) favorece pesos inv-RMSE. shrinkage 0.25 para n_windows=3 reduz overfit. Median como fallback robusto."
}
```

---

## 6. O Gatilho do Debate

Após o `Proposer` montar sua lista, o pipeline realiza uma **avaliação preliminar determinística** antes de decidir se vai convocar o `Skeptic` e o `Statistician`.

### O que acontece

O sistema executa todas as estratégias selecionadas pelo Proposer contra as janelas de validação históricas e calcula o `score` de cada uma. Isso gera um ranking provisório.

Em seguida, ele verifica:

```python
s1 = ranking[0]["score"]   # melhor candidato
s2 = ranking[1]["score"]   # segundo melhor

debate_margin_top2 = s2 - s1
```

### As 3 condições que ativam o Debate

| Condição | Trigger | Descrição |
|---|---|---|
| `debate=True` na chamada do pipeline | `"forced"` | O usuário forçou explicitamente |
| `force_debate=True` no JSON do Proposer | `"proposer_forced"` | O Proposer detectou instabilidade |
| `debate_margin_top2 < effective_debate_margin` (padrão `0.02`) | `"auto_margin"` | Os dois melhores estão perigosamente próximos |

O `effective_debate_margin` é calculado como:

```python
effective_debate_margin = max(debate_margin_parametro, proposer_debate_margin)
# ambos limitados ao intervalo [0.0, 0.1]
```

**Se o score da 1ª estratégia é `0.2300` e o da 2ª é `0.2312`**, a diferença é `0.0012`, que é menor que o limiar `0.02` → debate é acionado automaticamente (`"auto_margin"`).

**Por que acionar o debate nesses casos?** Porque uma diferença tão pequena pode ser ruído de variância, não evidência real de superioridade. O Skeptic e o Statistician têm a chance de revisar a lista e mudar o panorama antes da decisão final.

---

## 7. Agente 3 — Skeptic/Auditor

### O que ele é

O `Skeptic` é o **guardião da integridade matemática**. Ele não propõe nada — ele *filtra*. Sua função é remover estratégias perigosas e garantir diversidade mínima de tipos no conjunto candidato.

Temperatura: `0.2`.

### O que ele faz (Passo a Passo)

**1. Chama a tool `debate_packet`**

Essa ferramenta prepara um pacote com:

```json
{
  "candidate_ranking_top": [
    {"name": "inv_rmse_weights_per_horizon_k3_shrink02", "score": 0.2300, "type": "weighted"},
    {"name": "best_per_horizon_rolling", "score": 0.2312, "type": "selection"},
    {"name": "topk_mean_per_horizon_k3", "score": 0.2451, "type": "selection"},
    {"name": "baseline_median", "score": 0.2780, "type": "baseline"}
  ],
  "validation_summary": {
    "n_windows": 3,
    "n_models": 4,
    "disagreement": {"relative_spread_mean": 0.31}
  },
  "universe": {
    "candidate_names": ["baseline_mean", "baseline_median", "trimmed_mean_20", "best_single_rolling", ...],
    "leaderboards": {
      "RMSE": [
        {"name": "inv_rmse_weights_per_horizon_k3_shrink02", "aggregate_RMSE": 4.1},
        {"name": "best_per_horizon_rolling", "aggregate_RMSE": 4.5}
      ]
    }
  }
}
```

**2. Raciocina e checa os seguintes problemas**

- **Leakage**: Alguma estratégia usa `y_true` da janela que está sendo prevista? (Baselines como `mean` e `median` nunca vazam. Estratégias com `learns_weights=True` que não usam `_train_slice` poderiam vazar — mas todas as implementadas no código usam corretamente.)  
- **Homogeneidade**: O conjunto inteiro é só `baseline`? Se sim, adiciona pelo menos 1 candidato de tipo `weighted` ou `selection` do `universe.leaderboards`.
- **Redundância**: Há 3 variantes quase idênticas de `trimmed_mean`? Remove as mais fracas, mantém apenas 1.
- **Diversidade mínima**: O conjunto deve ter ao menos 2 tipos distintos.

**3. Retorna JSON — Output Final do `Skeptic`**

```json
{
  "add_names": [],
  "remove_names": ["trimmed_mean_20"],
  "params_overrides": {},
  "rationale": "trimmed_mean_20 é redundante com baseline_median e tem score pior (0.2780). Conjunto já tem 2 tipos (weighted + selection). Sem leakage detectado.",
  "changes": ["removed trimmed_mean_20 (redundant baseline)"],
  "when_good": "Quando o conjunto tiver ao menos 1 weighted e 1 selection com scores diferentes, nenhuma ação é necessária."
}
```

---

## 8. Agente 4 — Statistician

### O que ele é

O `Statistician` é o **especialista em robustez estatística**. Diferente do Skeptic que remove problemas, o Statistician *adiciona* proteções e substitui candidatos fracos por variações mais robustas.

Temperatura: `0.2`.

### O que ele faz (Passo a Passo)

**1. Chama a tool `debate_packet`** (mesma que o Skeptic, mas analisa diferente)

**2. Raciocina com base em evidências quantitativas**

Ele usa uma tabela de conhecimento embutida no seu prompt:

| Condição Observada | Ação do Statistician |
|---|---|
| `rmse_spread_ratio ≥ 0.3` | Adiciona `inverse_rmse_weights` ou `topk_mean` com k pequeno |
| `n_unique_winners ≥ 3` | Adiciona `best_per_horizon_by_validation` |
| `relative_spread_mean ≥ 0.25` | Adiciona `dba_combination` ou `trimmed_mean` com `trim_ratio=0.2` |
| `RMSE_std/RMSE > 0.3` (instabilidade) | Aumenta `shrinkage`; usa `robust_median` |
| `n_windows ≥ 6` | `ridge_stacking` torna-se viável — adiciona |
| Conjunto atual só tem baselines | **OBRIGATÓRIO** adicionar ≥1 weighted ou selection |

Ele também se baseia em referências acadêmicas:
- **DBA** (Petitjean et al., 2011): robusto quando modelos têm defasagens de fase.
- **Inverse-RMSE weights** (Timmermann, 2006): supera pesos uniformes quando a qualidade dos modelos varia.
- **Ridge stacking** (Gaillard & Goude, 2015): ótimo quando `n_windows ≥ 2 × n_models`.
- **Top-k mean** (M4 Competition, 2020): reduz variância de modelos outliers; `k = sqrt(n_models)` é um bom default.

**3. Retorna JSON — Output Final do `Statistician`**

```json
{
  "add_names": ["ridge_stacking_per_horizon_l2_10"],
  "remove_names": [],
  "params_overrides": {
    "inv_rmse_weights_per_horizon_k3_shrink02": {
      "shrinkage": 0.35,
      "top_k": 2
    }
  },
  "rationale": "n_windows=3 é baixo para ridge, mas relative_spread=0.31 justifica tentar com l2 alto. Aumentei shrinkage de 0.25 para 0.35 e reduzi top_k para 2 porque n_models=4 com n_windows=3 é limite.",
  "changes": [
    "added ridge_stacking_per_horizon_l2_10 (high spread justifies stacking)",
    "increased shrinkage to 0.35 for inv_rmse (low n_windows)"
  ],
  "when_good": "Conjunto diverso com weighted + selection + stacking. Nenhuma ação adicional necessária."
}
```

---

## 9. O Orquestrador Determinístico

Após todos os agentes terem falado, o Orquestrador executa a **avaliação final puramente matemática** — sem LLM, sem probabilidade, sem alucinação. É o juiz imparcial.

### Como ele avalia cada candidata

Para cada estratégia candidata, o sistema simula como ela teria se saído em **cada janela de validação histórica**, usando apenas dados anteriores àquela janela:

```
Janela i=0: sem dados anteriores → fallback para mean
Janela i=1: traineia com dados de i=0, prediz i=1
Janela i=2: traineia com dados de i=0 e i=1, prediz i=2
Janela i=3: traineia com dados de i=0, i=1, i=2, prediz i=3
```

Para cada janela avaliada, computa-se as métricas contra o `y_true` daquela janela.

### O Ranking Final

O resultado é um vetor de scores. O candidato com o **menor composite score** ganha.

Exemplo de ranking:

```json
{
  "ranking": [
    {"name": "inv_rmse_weights_per_horizon_k3_shrink02", "score": 0.2205, "aggregate": {"RMSE": 4.1, "SMAPE": 3.8, "MAPE": 3.6, "POCID": 81.2}},
    {"name": "ridge_stacking_per_horizon_l2_10",          "score": 0.2318, "aggregate": {"RMSE": 4.4, "SMAPE": 4.0, "MAPE": 3.9, "POCID": 79.5}},
    {"name": "best_per_horizon_rolling",                   "score": 0.2451, "aggregate": {"RMSE": 4.7, "SMAPE": 4.3, "MAPE": 4.1, "POCID": 78.1}},
    {"name": "baseline_median",                            "score": 0.2780, "aggregate": {"RMSE": 5.2, "SMAPE": 4.9, "MAPE": 4.7, "POCID": 74.3}}
  ],
  "best": {"name": "inv_rmse_weights_per_horizon_k3_shrink02", "score": 0.2205}
}
```

A estratégia vencedora (`inv_rmse_weights_per_horizon_k3_shrink02`) é então **aplicada à janela final** (o dado cego) para produzir a previsão real do sistema.

---

## 10. As Métricas de Avaliação — Por Que Cada Uma?

Esta seção explica de onde cada métrica vem, o que ela mede e por que ela está aqui.

### RMSE — Root Mean Squared Error

$$\text{RMSE} = \sqrt{\frac{1}{H} \sum_{h=1}^{H} (\hat{y}_h - y_h)^2}$$

**O que mede**: O erro médio entre previsão e valor real, com punição extra para erros grandes (por causa do quadrado).  
**Por que usamos**: RMSE é a métrica mais clássica de forecasting. Ela faz sentido quando erros grandes são especialmente ruins — por exemplo, prever 200 quando o real é 100 deve ser punido muito mais do que prever 105.  
**Limitação**: Ela é sensível à escala da série. Não tem sentido comparar o RMSE de uma série de temperatura (0–40°C) com uma série de preço de imóvel (R$ 100k–1M).

### SMAPE — Symmetric Mean Absolute Percentage Error

$$\text{SMAPE} = \frac{100\%}{H} \sum_{h=1}^{H} \frac{2|\hat{y}_h - y_h|}{|\hat{y}_h| + |y_h|}$$

**O que mede**: Erro percentual médio, mas com simetria entre superestimação e subestimação.  
**Por que usamos**: Diferente do MAPE regular, o SMAPE não explode quando `y_real ≈ 0` (pois usa a média do absoluto das duas). É muito mais estável para séries que passam perto de zero, como vendas com períodos de entressafra.  
**Por que é "simétrico"**: Se você previu 100 quando o real é 50, e se você previu 50 quando o real é 100 — o SMAPE pena os dois casos **da mesma forma**. O MAPE regular pena muito mais o segundo caso.

### MAPE — Mean Absolute Percentage Error

$$\text{MAPE} = \frac{100\%}{H} \sum_{h=1}^{H} \frac{|\hat{y}_h - y_h|}{|y_h| + \epsilon}$$

**O que mede**: Erro percentual absoluto relativo ao valor real.  
**Por que usamos**: Empresas e tomadores de decisão entendem porcentagem melhor do que unidades absolutas. Um MAPE de 5% significa que, em média, erramos 5% do valor real.  
**Por que temos o $\epsilon$**: Para evitar divisão por zero quando `y_real = 0`. O sistema usa `mape_zero="skip"` para simplesmente ignorar esses pontos no cálculo.

### POCID — Percentage of Correct Interval Direction

$$\text{POCID} = \frac{100\%}{H-1} \sum_{h=2}^{H} \mathbb{1}[(\hat{y}_h - \hat{y}_{h-1}) \cdot (y_h - y_{h-1}) > 0]$$

**O que mede**: A porcentagem de vezes em que a previsão acertou a **direção** da variação (subiu quando real subiu, ou desceu quando real desceu).  
**Por que usamos**: Para muitas aplicações (trading, gestão de estoque, planejamento de pico), acertar a direção é mais importante do que acertar o valor exato. Um modelo que sempre prevê o valor certo mas diz que a série vai subir quando na verdade vai cair é inútil.  
**Interpretação**: POCID = 100% significa que o modelo sempre acertou a direção. POCID = 50% é equivalente a jogar uma moeda. POCID < 50% significa que o modelo é sistematicamente enganoso.

---

## 11. Composite Score — A Fórmula do Ranking Final

Nenhuma métrica sozinha é suficiente. Uma estratégia pode ter RMSE ótimo mas POCID terrível. O sistema usa um **Composite Score** que combina as 4 métricas, normalizadas contra a linha de base (`baseline_mean`):

$$S = w_r \cdot \frac{\text{RMSE}_c}{\text{RMSE}_{base}} + w_s \cdot \frac{\text{SMAPE}_c}{\text{SMAPE}_{base}} + w_m \cdot \frac{\text{MAPE}_c}{\text{MAPE}_{base}} - w_p \cdot \frac{\text{POCID}_c}{100}$$

Onde o subscrito $c$ indica o candidato sendo avaliado e $base$ é a média simples de todos os modelos.

**Normalização pelo baseline**: Se `RMSE_candidato / RMSE_baseline = 0.85`, isso significa que o candidato é 15% melhor que a média simples em RMSE. Um valor normalizado < 1.0 é bom para RMSE, SMAPE, MAPE. Para POCID, maior é melhor (por isso subtrai no score).

**Quanto menor o score, melhor o candidato.**

### Presets de Pesos

O Proposer pode selecionar um preset de acordo com o tipo de série:

| Preset | $w_r$ (RMSE) | $w_s$ (SMAPE) | $w_m$ (MAPE) | $w_p$ (POCID) | Quando usar |
|---|---|---|---|---|---|
| `balanced` | 0.30 | 0.30 | 0.20 | 0.20 | Séries sem característica dominante |
| `rmse_focus` | 0.50 | 0.20 | 0.20 | 0.10 | Séries com erros grandes problemáticos |
| `direction_focus` | 0.25 | 0.25 | 0.10 | 0.40 | Séries onde direção é primordial (ex: trading) |
| `robust_smape` | 0.20 | 0.50 | 0.10 | 0.20 | Séries com valores próximos de zero |

---

## 12. Estratégias de Combinação e Pesos (Matemática Completa)

Abaixo estão todas as estratégias disponíveis na biblioteca, sua matemática e seus parâmetros configuráveis.

### Grupo 1: Baselines (sem aprendizado de pesos)

#### `mean`
$$\hat{y}_h = \frac{1}{M} \sum_{m=1}^{M} \hat{y}_{m,h}$$
Média simples de todos os modelos. Benchmark mínimo. Funciona bem quando os modelos têm erros parecidos e não há outlier.

#### `median`
$$\hat{y}_h = \text{median}(\hat{y}_{1,h}, ..., \hat{y}_{M,h})$$
Mais robusta a outliers extremos do que a média.

#### `trimmed_mean` (parâmetro: `trim_ratio`)
Remove os $\lfloor M \times \alpha \rfloor$ modelos de cada extremo (os piores e os melhores absolutos) e calcula a média do restante. Com `trim_ratio=0.2` e 10 modelos, remove 2 do topo e 2 do fundo, fazendo a média dos 6 centrais.

---

### Grupo 2: Seleção Pura (sem pesos — aprende por RMSE passado)

Todas as estratégias abaixo usam `_train_slice(i)` — ao decidir para a janela `i`, aprendem apenas com as janelas `0` até `i-1`.

#### `best_single` (seleção de 1 modelo global)
Para cada janela $i$:
1. Calcula o RMSE médio de cada modelo nas janelas anteriores: $\text{RMSE}_m = \sqrt{\frac{1}{(i)(H)} \sum_{j<i} \sum_h (\hat{y}_{m,j,h} - y_{j,h})^2}$
2. Seleciona o modelo com menor RMSE: $m^* = \arg\min_m \text{RMSE}_m$
3. Usa exclusivamente as predições de $m^*$

Risco: se o melhor modelo histórico tiver performance flutuante, a seleção pode ser instável.

#### `best_per_horizon` (seleção por horizonte)
Como o `best_single`, mas faz a seleção **separadamente para cada passo do horizonte** $h$:

$$m^*(h) = \arg\min_m \sqrt{\frac{1}{i} \sum_{j<i} (\hat{y}_{m,j,h} - y_{j,h})^2}$$

Se `n_unique_winners ≥ 3`, significa que modelos diferentes são melhores em horizontes diferentes — esse método captura isso.

#### `topk_mean_per_horizon` (parâmetro: `top_k`)
Como o `best_per_horizon`, mas em vez de usar 1 modelo, usa a **média dos `top_k` melhores** por horizonte:
$$\hat{y}_h = \frac{1}{k} \sum_{m \in \text{Top-}k(h)} \hat{y}_{m,h}$$
Reduz a variância de seleção única. Recomendado `k = sqrt(M)` ou simplesmente `k=2` quando `n_windows` é pequeno.

---

### Grupo 3: Pesos Proporcionais ao Erro Passado

#### `inverse_rmse_weights_per_horizon` (parâmetros: `top_k`, `shrinkage`, `eps`)

**Passo 1**: Calcula o RMSE de cada modelo por horizonte nas janelas anteriores.

**Passo 2**: Inverte o erro para dar mais peso aos melhores modelos:
$$w_m^{\text{raw}}(h) = \frac{1}{\text{RMSE}_m(h) + \epsilon}$$

**Passo 3** (opcional, parâmetro `shrinkage` $S \in [0, 0.9]$): move os pesos em direção à distribuição uniforme para evitar confiança excessiva num único modelo:
$$w_m(h) = (1 - S) \cdot \frac{w_m^{\text{raw}}(h)}{\sum_j w_j^{\text{raw}}(h)} + S \cdot \frac{1}{M}$$

**Passo 4**: Projeção no simplex (ver seção 13) para garantir $w_m \geq 0$ e $\sum w_m = 1$.

**Passo 5**: Combina:
$$\hat{y}_h = \sum_{m=1}^{M} w_m(h) \cdot \hat{y}_{m,h}$$

**Com `top_k`**: Antes de computar pesos, filtra apenas os `top_k` modelos por RMSE, zerando os demais.

#### `exp_weighted_average_per_horizon` (parâmetros: `eta`, `trim_ratio`)

Os erros decaem exponencialmente:
$$w_m(h) \propto \exp\left(-\eta \cdot \text{RMSE}_m(h)\right)$$

- Com `eta=1.0`: modelo com RMSE 2x maior recebe peso ~7x menor.  
- Com `eta=2.0`: punição é muito mais severa (quadrática no expoente).  
- Com `trim_ratio`: antes de calcular os pesos, descarta os `(1 - trim_ratio)` piores modelos.

#### `poly_weighted_average_per_horizon` (parâmetros: `power`, `trim_ratio`)

Versão algébrica em vez de exponencial:
$$w_m(h) \propto \left(\text{RMSE}_m(h) + \epsilon\right)^{-\text{power}}$$

Com `power=1` equivale a `inverse_rmse`. Com `power=2`, os melhores modelos dominam ainda mais.

#### `ade_dynamic_error_per_horizon` (parâmetros: `beta`, `eta`, `trim_ratio`)

Versão adaptativa com memória exponencial. Em vez de calcular o RMSE acumulado do passado, usa uma **Média Móvel Exponencial (EMA) dos erros mais recentes**:

$$\text{EMA\_err}_{m,h}^{(i)} = \beta \cdot |\hat{y}_{m,i-1,h} - y_{i-1,h}| + (1 - \beta) \cdot \text{EMA\_err}_{m,h}^{(i-1)}$$

$$w_m(h) \propto \exp\left(-\eta \cdot \text{EMA\_err}_{m,h}^{(i)}\right)$$

Com `beta=0.5`, os dois últimos erros respondem por ~75% da memória. Isso faz o sistema reagir rapidamente a mudanças de regime (concept drift).

---

### Grupo 4: Stacking (regressão sobre os preditores)

#### `ridge_stacking_per_horizon` (parâmetros: `l2`, `top_k`)

É o método mais sofisticado. Em vez de calcular pesos por heurística, **faz uma regressão linear** das predições passadas contra os valores reais passados:

**Configura o problema de regressão** (para cada horizonte $h$, com janelas de treino $j < i$):
- $X \in \mathbb{R}^{(i) \times M}$: matriz onde cada linha é uma janela e cada coluna é um modelo
- $y \in \mathbb{R}^{i}$: vetor de valores reais daquelas janelas no horizonte $h$

**Resolve o sistema Ridge** (Mínimos Quadrados com Penalidade L2):
$$w^* = (X^TX + \lambda I)^{-1} X^T y$$

onde $\lambda$ = parâmetro `l2` (valores: 0.1–1000). Um $\lambda$ alto força os pesos a serem pequenos e próximos entre si — útil quando os modelos são colineares (produzem previsões parecidas).

**Projeção no Simplex**: O vetor $w^*$ pode ter valores negativos (permitidos matematicamente pelo Ridge). Após rodar o Ridge, os pesos são projetados no simplex para forçar $w \geq 0$ e $\sum w = 1$.

**Com `top_k`**: Antes de rodar a regressão, pré-filtra os `top_k` modelos por RMSE histórico, reduzindo a dimensionalidade.

**Quando usar**: Ridge Stacking é viável quando `n_windows ≥ 2 × n_models`. Com apenas 3 janelas e 5 modelos, o sistema está sub-determinado — o `l2` alto age como regularização para evitar overfitting.

---

### Grupo 5: DBA — DTW-Barycenter Averaging

#### `dba` (parâmetros: `top_k`, `max_iter`)

O DBA não é uma média ponderada — é uma **média no espaço das trajetórias com alinhamento temporal flexível (Dynamic Time Warping)**.

Imagine que dois modelos previram o mesmo padrão de pico, mas um previu o pico em `t+3` e o outro em `t+5`. A média simples daria um "vale" artificial entre os dois picos. O DBA encontra a trajetória centróide que minimiza a **soma das distâncias DTW** a todas as previsões.

- Iterativo (parâmetro `max_iter`): em cada iteração, alinha cada previsão à estimativa do centróide atual e atualiza o centróide.
- Com `top_k`: só usa os `top_k` modelos com menor RMSE histórico antes de calcular o centróide.
- Implementação: usa `tslearn.barycenters.dtw_barycenter_averaging`.

**Quando usar**: Alta velocidade de divergência entre modelos (`relative_spread_mean ≥ 0.25`) ou quando modelos capturaram o mesmo padrão mas com defasagens de fase diferentes.

---

## 13. Proteção Matemática: Projeção no Simplex

Todas as estratégias que aprendem pesos usam a função `_project_simplex(v)` como última etapa antes de combinar as predições.

### O que é o Simplex?

O conjunto simplex de probabilidade é definido como:
$$\Delta = \{w \in \mathbb{R}^M : w_m \geq 0 \text{ para todo } m, \text{ e } \sum_{m=1}^M w_m = 1\}$$

### Por que projetar?

- O Ridge pode retornar pesos negativos (é matematicamente válido mas economicamente absurdo: "subtrair" a previsão de um modelo).
- O `inverse_rmse` sem normalização pode somar mais de 1.0 por erro numérico.
- O LLM pode sugerir `shrinkage=1.5` que distorceria os pesos.

A projeção garante que:
1. **Nenhum modelo tem peso negativo** — o pior que pode acontecer é peso = 0.
2. **Os pesos somam exatamente 1.0** — a combinação é uma média ponderada válida.

### O Algoritmo

```python
def _project_simplex(v):
    u = sort(v, descending=True)
    cssv = cumsum(u)
    rho = last index where: u[j] * (j+1) > (cssv[j] - 1)
    theta = (cssv[rho] - 1) / (rho + 1)
    return max(v - theta, 0)  # shift e clip
```

---

## 14. Previsão Final: da Melhor Estratégia ao Número

Após o ranking ser calculado e a melhor estratégia ser selecionada, o sistema aplica essa estratégia **à janela final** — o dado cego que nunca foi visto durante toda a discussão.

O processo é idêntico ao da avaliação, mas usados **todos os dados de validação** como treino (o maior `_train_slice` disponível). A estratégia agora aprende com tudo que tem disponível e produz a combinação final das previsões dos modelos base para o horizonte de teste.

```
Output final: vetor de floats com comprimento = horizonte de previsão
Ex: [105.2, 107.1, 109.4, 111.8, 113.0]
```

Esse vetor é o resultado final do sistema — a melhor estimativa combinada de todos os modelos base, escolhida pelo pipeline multi-agente e validada deterministicamente contra o passado observado.

