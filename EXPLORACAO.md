# EXPLORACAO.md — Mapeamento do orquestrador atual (Passo 1)

> Documento produzido **antes de qualquer alteração de código**, conforme exigido pelo Passo 1 da
> especificação de reescrita para arquitetura de Agente Combinador via ReAct.
>
> **Estado do repositório explorado:** branch `v1_proposta`, HEAD `22e41be6` ("timeseries").
> Working tree limpo exceto por `.claude/` e `.vscode/` (não rastreados).

---

## 0. Divergências encontradas — ler antes de implementar

A especificação pressupõe algumas coisas que **não batem** com o código. Nenhuma delas invalida a
proposta, mas todas mudam decisões de implementação. Estão listadas aqui, em ordem de impacto.

### D1 — A branch explorada não é a que o contexto da sessão indicava

O snapshot de git que abriu a sessão descrevia a branch `halmoc` com arquivos
`orchestrator/halmoc_pipeline.py`, `orchestrator/conformal.py`, `orchestrator/mcs.py`,
`orchestrator/meta_combiner.py`, `orchestrator/memory.py`, `orchestrator/feature_extractor.py` e
`orchestrator_langchain/prompts/{council_member,diagnostician,judge}.md`. **Nenhum desses arquivos
existe na branch atual (`v1_proposta`).** Eles vivem em `halmoc` (branch local e remota, existentes).
Existe também `V3_CONTRIBUTION.md` na raiz descrevendo uma arquitetura V3
(`SeriesAnalyst → ModelCritic → CombinationArchitect`) que igualmente **não está implementada aqui**.

Toda a exploração abaixo é sobre `v1_proposta`. Se a reescrita deve partir de `halmoc` (que já tem
MCS, conformal e meta-combiner), isso precisa ser decidido antes do Passo 2 — parte do que a
especificação pede como "novo" pode já existir lá.

### D2 — O sistema de debate Proposer/Skeptic/Statistician/PatternAnalyst **existe mesmo**

Confirmado por leitura, não presumido. Os quatro papéis estão em
[orchestrator_langchain/agents.py:103-144](orchestrator_langchain/agents.py#L103-L144), com prompts
em [orchestrator_langchain/prompts/](orchestrator_langchain/prompts/) e orquestração em
[orchestrator/pipeline.py:642-1305](orchestrator/pipeline.py#L642-L1305). Existe ainda um quinto
papel, **Orchestrator** ([agents.py:147](orchestrator_langchain/agents.py#L147) +
[prompts/orchestrator.md](orchestrator_langchain/prompts/orchestrator.md)), que **não é usado** pelo
pipeline em produção (a seleção final é determinística, feita em código). Só é referenciado por
`orchestrator/run_research_loop.py`, que é código morto (ver D8).

### D3 — `serie_treino`: origem **validada empiricamente** (decisão tomada)

> **Atualização após decisão do usuário.** A série de treino virá do arquivo original, fora do repo,
> em `/home/lucas/Documents/mestrado/forecasting_datasets/` (uma pasta acima da raiz, i.e.
> `../forecasting_datasets`). O `exec_dataset_orchestrator` receberá um parâmetro novo com o nome do
> arquivo-fonte. **Antes de escrever qualquer função, o mapeamento `dataset_index` → série foi
> validado ponto a ponto** contra os CSVs de resultado já existentes. Resultado abaixo.

#### Método da validação

Para cada dataset, para cada `dataset_index`, para as 4 janelas mais recentes, comparei a coluna
`test` do CSV de resultados com `serie[len - (k+1)*H : len - k*H]` extraída do arquivo-fonte
(`k=0` é o teste final). Tolerância `rtol=1e-4`.

#### Resultado — mapeamento confirmado

**Formato canônico: `.tsf` para todos os datasets** (decisão do usuário — é o formato usado para
gerar as previsões dos modelos individuais). `base/mes_11_venda_mensal.tsf` foi copiado para
`../forecasting_datasets/mes_11_venda_mensal.tsf` para uniformizar o `source_dir`.

| Dataset (resultados) | Arquivo-fonte | Pré-processamento obrigatório | `dataset_index` → série | Casos conferidos |
|---|---|---|---|---|
| `ANP_MONTHLY` | `mes_11_venda_mensal.tsf` | filtro `should_remove(window=24)`: **216 → 182** séries | posição no df **filtrado** | **728/728** ✅ |
| `NN5_WEEKLY_DATASET` | `nn5_weekly_dataset.tsf` | nenhum | posição direta | **444/444** ✅ |
| `M4_WEEKLY_DATASET` | `m4_weekly_dataset.tsf` | nenhum | posição direta | **1436/1436** ✅ |
| `ETTH1` | `ETTh1.tsf` | nenhum | posição direta | ⚠ ver nota abaixo |
| `ETTH2` | `ETTh2.tsf` | nenhum | posição direta | ⚠ ver nota abaixo |
| `ETTM1` | `ETTm1.tsf` | `frequency=half_hourly` → `30min` | posição direta | ⚠ ver nota abaixo |
| `ETTM2` | `ETTm2.tsf` | `frequency=half_hourly` → `30min` | posição direta | ⚠ ver nota abaixo |

#### ⚠ Nota sobre os `.tsf` do ETT **nesta máquina**

Os arquivos `ETTh1.tsf`, `ETTh2.tsf`, `ETTm1.tsf` e `ETTm2.tsf` presentes em
`/home/lucas/Documents/mestrado/forecasting_datasets/` **nesta máquina** têm **1 série cada** (foram
regravados em 28/12/2025 contendo só a variável OT — o cabeçalho do arquivo diz isso). Os CSVs de
resultado têm **7 séries** por dataset ETT, que é o que a versão original do `.tsf` (7 variáveis:
`HUFL, HULL, MUFL, MULL, LUFL, LULL, OT`) fornece.

Confirmei que os valores batem: reconstruindo as 7 séries a partir do `ETTh1.csv`/`ETTh2.csv`
(mesma pasta) o casamento é **28/28** para ETTh, e **28/28** para ETTm aplicando
`resample("30min").mean()` — o que prova que os resultados vieram do `.tsf` de 7 séries, com o ETTm
em frequência `half_hourly`. Ou seja: **o `.tsf` correto existe, só não é o que está nesta máquina.**

Como a execução será na outra máquina, o loader lê `.tsf` normalmente. O guarda-corpo de runtime
(abaixo) detecta e reporta com mensagem clara caso o `.tsf` encontrado tenha menos séries do que os
resultados exigem, em vez de casar a série errada em silêncio.

#### Duas armadilhas que a validação evitou

1. **O `.tsf` do ANP correto não é o óbvio.** `monthly_fuel_sales_by_state.tsf` (raiz do repo, 216
   séries, vai até 2025-06) **não bate** — é 7 meses mais longo. O que bate é
   `base/mes_11_venda_mensal.tsf` (216 séries, até 2024-11). Ambos têm 216 séries e o mesmo
   `series_name`/`state_code`/`product` no índice 0, então a confusão passaria despercebida por
   qualquer checagem de metadados — só a comparação de valores pega.
3. **O ANP passa por um filtro.** `dataset_index` **não é** a posição no `.tsf` original: 216 séries
   viram 182 após `should_remove(window_size=24)` (remove séries com >50% de zeros em alguma janela
   de 24 meses). A função está comentada em
   [run_tsf_regressors.py:820-828](run_tsf_regressors.py#L820-L828) e reproduz o corte com precisão
   exata (216 → 182, confere com o `nunique()` dos resultados).

#### Como a série é reconstruída (idêntico ao pipeline de geração)

Confirmado em [run_tsf_regressors.py:614-619](run_tsf_regressors.py#L614-L619) e
[:638-641](run_tsf_regressors.py#L638-L641):

```python
freq_map = {"yearly":"Y","monthly":"M","weekly":"W","daily":"D",
            "hourly":"H","half_hourly":"30min","15min":"15min"}
index_series = pd.date_range(start=start_timestamp, periods=len(series_value), freq=freq)
series = pd.Series(series_value, index=index_series)
# janela k (k=0 é o teste final):
train, test = aux_series[:-horizon], aux_series[-horizon:]   # peeling sucessivo
```

Portanto **`serie_treino` para o teste final = `series[:-horizon]`**. Tamanhos reais:
`ANP_MONTHLY` 419−12 = **407 pontos**; `ETTH1` 17420−24 = **17396 pontos**;
`NN5_WEEKLY` e `M4_WEEKLY` idem. Ou seja, a preocupação de "só 72 pontos" levantada antes **some
completamente** — o `series_profile()` terá histórico de sobra para STL, ADF/KPSS e catch22.

#### Guarda-corpo obrigatório na implementação

O loader **não** vai confiar no mapeamento cegamente. Para cada série carregada, ele verifica que
`serie[-horizon:]` é igual (dentro de tolerância) à coluna `test` da última linha do CSV de
resultados daquele `dataset_index`, e **levanta exceção** se divergir. Isso torna impossível
combinar previsões de uma série com o perfil de outra — que é exatamente o risco apontado.

---

### D3-bis — Nota histórica: o estado anterior à decisão acima

A Seção 3.1 da especificação lista `serie_treino` (série histórica completa) como entrada, e a
ferramenta `series_profile()` (3.4.1) precisa dela para STL, ADF/KPSS e catch22. **O orquestrador
atual nunca vê a série histórica.** Ele consome exclusivamente os CSVs de previsão por modelo
(`timeseries/mestrado/resultados/<MODELO>/normal/<DATASET>.csv`). Os arquivos `.tsf` originais dos
datasets ETT/M4/NN5 **não estão no repositório** (só existem `monthly_fuel_sales_by_state.tsf` e
`base/mes_11_venda_mensal.tsf`, ambos ANP).

**Descartada** (mantida aqui só como registro): reconstruir a série concatenando a coluna `test` das
janelas anteriores. Funciona — as janelas são contíguas e não se sobrepõem — mas daria apenas 72
pontos em `ETTH1` contra os 17396 disponíveis na fonte original. Superada pela solução validada em
D3.

### D4 — São **2** janelas de validação na configuração atual, não 3

[run_tsf_orchestrator.py:684](run_tsf_orchestrator.py#L684) passa `train_window=3`, e
[orchestrator_langchain/context.py:103](orchestrator_langchain/context.py#L103) fatia com
`df.iloc[-train_window:-1]` → `iloc[-3:-1]` = **2 janelas**. Para obter as 3 janelas que a
especificação assume (e que os baselines já calculados em disco usam), é preciso `train_window=4`.

Note o contraste: [combinations/aux.py:52-55](combinations/aux.py#L52-L55) pega `iloc[-2]`, `iloc[-3]`
e `iloc[-4]` explicitamente — ou seja, **as baselines `mean`/`median`/`dba`/`ADE`/`FFORMA` já em disco
usaram 3 janelas**, enquanto o orquestrador LLM v1 usou 2. Qualquer comparação linha a linha entre o
orquestrador e essas baselines na análise do TCC está, hoje, comparando protocolos diferentes.

### D5 — O pool tem 19 modelos ativos, mas **25 estão disponíveis** para `ETTH1`

[run_tsf_orchestrator.py:644-671](run_tsf_orchestrator.py#L644-L671) lista 19 modelos ativos e 6
comentados (as variantes `ridge`). Conferindo o disco, para `ETTH1` existem resultados para
exatamente **25 modelos base**: `ARIMA, ETS, THETA, ridge, rf, catboost, CWT_ridge, CWT_rf,
CWT_catboost, DWT_ridge, DWT_rf, DWT_catboost, FT_rf, FT_catboost, ONLY_CWT_ridge, ONLY_CWT_rf,
ONLY_CWT_catboost, ONLY_DWT_ridge, ONLY_DWT_rf, ONLY_DWT_catboost, ONLY_FT_ridge, ONLY_FT_rf,
ONLY_FT_catboost, NaiveSeasonal, NaiveMovingAverage`. Não há `FT_ridge`, nem `svr`, `NBEATS`,
`NaiveDrift` para este dataset. Descomentar as 6 variantes `ridge` fecha os "~25 modelos" da
especificação sem precisar treinar nada novo.

### D6 — As baselines de comparação **já estão calculadas em disco**

`timeseries/mestrado/resultados/{mean,median,dba,ADE,FFORMA}/<DATASET>.csv` (formato plano, 13
colunas, sem subpasta `normal/`). O campo novo `baseline_results_json` (Seção 4.4) pode ser
**preenchido por leitura desses arquivos**, em vez de recalcular ADE/FFORMA por série — o que é
importante porque nenhuma das duas é recalculável por série isoladamente (ver D7).

### D7 — `weights_softmax_neg_error` e `weights_feature_based` não existem na forma que a especificação sugere

A especificação diz "replica a fórmula do ADE que já deve existir em algum lugar do código atual" e
"reaproveite lógica existente se houver" (FFORMA). O que existe de fato:

| Ferramenta pedida (3.4.3) | O que existe hoje | Reaproveitável direto? |
|---|---|---|
| `weights_inverse_error` | `_weights_inverse_rmse` em [strategies.py:104](orchestrator/strategies.py#L104) e [final_predictor.py:44](orchestrator/final_predictor.py#L44) (duplicado) | **Sim** |
| `weights_softmax_neg_error` (ADE) | `_weights_exp` = `softmax(-η·erro)` em [strategies.py:86](orchestrator/strategies.py#L86); método `ade_dynamic_error_per_horizon` ([strategies.py:445](orchestrator/strategies.py#L445)) usa EMA de erro absoluto + esse softmax | **Sim, mas é uma reimplementação local**, não o ADE canônico |
| — | O ADE "de verdade" ([combinations/ade.py:270](combinations/ade.py#L270)) usa a lib `metaforecast`, roda **cross-series** e escreve CSV direto; não expõe função por série | Não |
| `weights_ols` | **Não existe.** O mais próximo é `_ridge_weights` (ridge + projeção no simplex) em [strategies.py:176](orchestrator/strategies.py#L176) | Só como base; OLS puro é código novo |
| `weights_feature_based` (FFORMA) | [combinations/fforma.py](combinations/fforma.py) treina XGBoost **sobre todas as séries do dataset** (meta-modelo cross-series). Fallback `_compute_weights_softmax` = `softmax(-SMAPE)` ([fforma.py:276](combinations/fforma.py#L276)) | **Não** por série; o fallback softmax sim |

Ou seja: `weights_feature_based` como *tool per-série* é **código novo**, e conceitualmente estranho
(FFORMA é meta-aprendizado cross-series por construção). Sugestão: implementá-la como um wrapper que
ou (a) treina o meta-modelo uma vez por dataset na Fase 0 e apenas *consulta* na Fase 3, ou (b) cai no
`softmax(-SMAPE)` por série. Decisão a tomar antes do Passo 6.2.

### D8 — Bugs reais no código atual (relevantes para a leitura dos resultados já gerados)

1. **Modelos de Skeptic e Statistician estão trocados.**
   [orchestrator/pipeline.py:642-647](orchestrator/pipeline.py#L642-L647) declara a ordem
   `(proposer, skeptic, statistician, pattern_analyst)`, mas
   [orchestrator_langchain/pipeline.py:29-34](orchestrator_langchain/pipeline.py#L29-L34) chama
   posicionalmente `(proposer_model, statistician_model, skeptic_model, pattern_analyst_model)`.
   **Todos os CSVs de `orchestrator_llm_v1_pattern` já gerados têm o Skeptic rodando com
   `qwen3:14b` e o Statistician com `gpt-oss:20b`**, o inverso do que
   [run_tsf_orchestrator.py:679-680](run_tsf_orchestrator.py#L679-L680) configura.

2. **`_log` indefinido em escopo de módulo.**
   [orchestrator/pipeline.py:166](orchestrator/pipeline.py#L166) chama `_log(...)` dentro de
   `_validate_actions_against_universe`, mas `_log` só é definido localmente dentro de
   `run_llm_pipeline` ([pipeline.py:666](orchestrator/pipeline.py#L666)). Se o caminho de resolução de
   nome desconhecido for atingido → `NameError`, que vira hard-stop da série.

3. **Fallbacks de tool são código morto.**
   [pipeline.py:719](orchestrator/pipeline.py#L719) e [pipeline.py:774](orchestrator/pipeline.py#L774)
   chamam `_build_fold_cot_context_tool.entrypoint()` / `_proposer_brief_tool.entrypoint()`, mas ambas
   são funções Python simples importadas de `orchestrator.tools` — não têm atributo `.entrypoint`.
   O `AttributeError` é engolido pelo `try/except`. Consequência: se o Proposer pular a tool, o
   fallback **não** funciona e a série morre no hard-stop de
   [pipeline.py:780](orchestrator/pipeline.py#L780).

4. **`run_research_loop.py` é código morto.**
   [orchestrator/run_research_loop.py:11](orchestrator/run_research_loop.py#L11) importa
   `from agent.context import ...` — o pacote `agent` não existe no repositório. O arquivo nunca roda.

### D9 — Não há CLI

A especificação pergunta pelos "parâmetros de CLI/config aceitos". **Não existe nenhum.** Todos os
parâmetros são hardcoded no bloco `if __name__ == "__main__"`
([run_tsf_orchestrator.py:643-688](run_tsf_orchestrator.py#L643-L688)): lista de modelos, dataset,
`use_llm`, os quatro `ModelConfig`, `rolling`, `train_window`, `llm_logs`. O parâmetro `version`
(default `"v1_pattern"`) é o que nomeia a pasta de saída. Os parâmetros `start_index`/`end_index`
estão comentados na assinatura. O objeto único de configuração por rodada pedido no Passo 5 é,
portanto, **construção nova** — não há nada para adaptar.

---

## 1. Mapa de chamadas a partir de `run_tsf_orchestrator.py`

```
run_tsf_orchestrator.py
└── exec_dataset_orchestrator(models, dataset, use_llm, 4× ModelConfig, ...)   [:132]
    ├── pd.read_csv("resultados/catboost/normal/<DATASET>.csv")                [:150]
    │     └── deriva: horizon, final_test, num_series (= dataset_index.nunique())
    ├── cria CSV de saída com COLS_SERIE / faz reindex se já existir           [:181-194]
    │
    └── for i in range(num_series):                                            [:196]
        ├── orchestrator_langchain.context.init_context()                      [:197]
        ├── CONTEXT_MEMORY["models_available"] = models                        [:198]
        ├── orchestrator_langchain.context.generate_all_validations_context()  [:199]
        │     └── read_model_preds(m, i, dataset) para cada modelo             [ctx:68]
        │           └── preenche CONTEXT_MEMORY["all_validations"] e ["predictions"]
        │
        ├── SE use_llm:  orchestrator_langchain.pipeline.run_langchain_pipeline()  [:211]
        │     ├── monkey-patch das 4 factories de agente em orchestrator.pipeline  [lc_pipe:24-27]
        │     └── orchestrator.pipeline.run_llm_pipeline(...)                      [lc_pipe:29]
        │           │  ⚠ ordem posicional troca skeptic↔statistician (ver D8.1)
        │           │
        │           ├── Fase A — PatternAnalyst                               [pipe:693-722]
        │           │     agente → tool build_fold_cot_context_tool()          [tools:1227]
        │           │       └── grava context["pattern_analyst_cot_context"]
        │           ├── Fase B — Proposer                                      [pipe:740-887]
        │           │     agente → tool proposer_brief_tool()                  [tools:42]
        │           │       ├── _build_validation_summary()                    [tools:229]
        │           │       ├── _candidate_universe_from_summary()             [tools:766]
        │           │       ├── _suggest_candidates_from_summary()             [tools:378]
        │           │       └── grava context["orchestrator_proposer_brief"]
        │           │     → _validate_actions_against_universe()               [pipe:122]
        │           │     → _apply_actions_to_payload()                        [pipe:291]
        │           ├── Fase C — avaliação pré-debate + gate estatístico       [pipe:912-993]
        │           │     evaluate_all()  +  diagnostics.tie_break_analysis()  [diag:375]
        │           ├── Fase D — Debate 2 rodadas Skeptic ↔ Statistician       [pipe:995-1156]
        │           │     ambos → tool build_debate_packet_tool()              [tools:1734]
        │           ├── Fase E — avaliação determinística final                [pipe:1206]
        │           │     evaluator.evaluate_all()                             [eval:131]
        │           │       └── evaluate_candidate() → strategies.generate_combined_predictions()
        │           └── Fase F — previsão final                                [pipe:1222]
        │                 final_predictor.predict_final_from_context()         [fp:103]
        │
        ├── SENÃO:      orchestrator.pipeline.run_deterministic_pipeline()     [:271, pipe:596]
        │                 (8 candidatos fixos DEFAULT_CANDIDATES → evaluate_all → predict_final)
        │
        ├── get_predictions_models(...) → obtém `test` real do horizonte final [:273]
        ├── extrai ~40 campos do `description` JSON e de `llm_artifacts`       [:283-534]
        ├── calcula mape/pocid/smape/rmse/msmape/mae                           [:555-577]
        ├── monta `data_serie` (o dicionário do Passo 4)                       [:579-633]
        └── append no CSV; se hard_stop → RuntimeError                         [:635-640]
```

**Relação entre as duas pastas** (Passo 1.3, confirmado por leitura, não presumido):
elas são **complementares, não versões alternativas**.

- `orchestrator/` = **núcleo determinístico**. Contrato de dados, métricas, estratégias de
  combinação, avaliador anti-leakage, diagnósticos estatísticos, preditor final, catálogo de tools
  determinísticas. Não depende de LangChain.
- `orchestrator_langchain/` = **camada LLM + ingestão**. Agentes LangChain sobre `ChatOllama`,
  prompts em markdown, wrappers `@tool` das tools determinísticas, e o `CONTEXT_MEMORY` global +
  leitores de CSV.
- O acoplamento é circular e por *monkey-patching*: `orchestrator/agents.py` **reexporta** as
  factories de `orchestrator_langchain.agents` (13 linhas, é só um alias), e
  `orchestrator_langchain/pipeline.py` sobrescreve essas mesmas factories em `orchestrator.pipeline`
  antes de chamá-lo. `orchestrator/tools.py` e `orchestrator/data_contract.py` importam
  `orchestrator_langchain.context`. Não há como usar uma pasta sem a outra hoje.

---

## 2. Agentes / papéis atuais e o que cada um faz

| # | Papel | Factory | Prompt | Tool única | Saída JSON esperada |
|---|---|---|---|---|---|
| 1 | **PatternAnalyst** | [agents.py:103](orchestrator_langchain/agents.py#L103) | `pattern_analyst.md` (93 l.) | `build_fold_cot_context` | `trend_champion`, `seasonality_champion`, `overall_champion`, `horizon_specialists`, `tier1/2_models`, `recommended_method_hint`, `cot_narrative` |
| 2 | **Proposer** | [agents.py:114](orchestrator_langchain/agents.py#L114) | `proposer.md` (60 l.) | `proposer_brief` | `selected_names`, `params_overrides`, `score_preset`, `force_debate`, `debate_margin`, `rationale` |
| 3 | **Skeptic** | [agents.py:125](orchestrator_langchain/agents.py#L125) | `skeptic.md` (44 l.) | `debate_packet` | `add_names`, `remove_names`, `params_overrides`, `rationale`, `changes`, `when_good` |
| 4 | **Statistician** | [agents.py:136](orchestrator_langchain/agents.py#L136) | `statistician.md` (48 l.) | `debate_packet` | idem Skeptic |
| 5 | **Orchestrator** *(não usado)* | [agents.py:147](orchestrator_langchain/agents.py#L147) | `orchestrator.md` (30 l.) | `evaluate_strategies` | `best_name`, `reasoning`, `top3`, `when_good`, `debate_notes` |

**Como o laço de decisão funciona hoje** (para contraste com o ReAct da Seção 3.3):

1. PatternAnalyst analisa decomposição STL dos folds → escolhe "campeões" de tendência/sazonalidade
   e emite um *hint* de método. Falha aqui é **não-fatal**.
2. Proposer recebe o brief determinístico (sumário de validação + biblioteca de candidatos + hint do
   PatternAnalyst) e **seleciona por whitelist** um subconjunto de estratégias candidatas + um
   *score preset*. Ele nunca escreve números de previsão — só nomes e knobs limitados.
3. O código avalia esse conjunto e roda um **gate estatístico** (Diebold-Mariano + bootstrap pareado,
   α=0.10). Se top-1 e top-2 são estatisticamente indistinguíveis → dispara o debate. Fallback:
   margem de score < 2%.
4. Debate: 2 rodadas. Rodada 1 = Skeptic e Statistician respondem cegos um ao outro. Rodada 2 = cada
   um vê o JSON do par e revisa. Ações são validadas contra o universo e **clampadas**
   (`ALLOWED_PARAM_EDITS = {top_k, trim_ratio, shrinkage, l2, period}`, `method` nunca muda).
5. Avaliação determinística final escolhe o melhor por *composite score*; `predict_final_from_context`
   aplica a estratégia vencedora às previsões de teste.

**O que já é compatível com os princípios da Seção 3.2** (e vale manter conceitualmente, ainda que
reimplementado): dados nunca entram crus no prompt (princípio 1) — as tools retornam sumários; o LLM
nunca gera número de previsão (princípio 2) — só nomes e knobs clampados; separação
diagnóstico/decisão (princípio 4) existe via PatternAnalyst; zero fine-tuning (princípio 9).

**O que não existe hoje e é o núcleo da proposta nova:** laço ReAct iterativo com histórico de
tentativas ranqueado (princípio 3), pré-semeadura de baselines no histórico (princípio 5), orçamento
de iterações com parada antecipada (princípio 8), *handles* de pesos (princípio 2, forma forte), e
justificativa causal obrigatória (princípio 7). O fluxo atual é **linear e de passe único**, não um
laço.

---

## 3. Ferramentas determinísticas já existentes

Todas em `orchestrator/tools.py`, expostas ao LangChain por wrappers em
`orchestrator_langchain/langchain_tools.py`. Todas gravam seu nome em `context["tools_called"]`.

| Tool | Linha | O que retorna | Reaproveitável em 3.4? |
|---|---|---|---|
| `proposer_brief_tool()` | [tools.py:42](orchestrator/tools.py#L42) | `validation_summary` + `candidate_library` + `recommended_knobs` + `score_presets` + insights do PatternAnalyst | **Não** — é a interface do Proposer. Mas `_build_validation_summary` interno sim |
| `build_fold_cot_context_tool()` | [tools.py:1227](orchestrator/tools.py#L1227) | STL por fold de `y_true` e de cada modelo, correlações trend/seas, RMSE early/late, diagnósticos A1/A2, tiers | **Parcialmente** — a decomposição STL e as correlações servem a `stl_summary()`; o resto é lógica do PatternAnalyst |
| `build_debate_packet_tool()` | [tools.py:1734](orchestrator/tools.py#L1734) | ranking top-N, margem top-2, `tie_break_analysis`, vencedores por horizonte, leaderboards do universo, knobs permitidos | **Não** — é a interface do debate |
| `evaluate_strategies_tool()` | [tools.py:1920](orchestrator/tools.py#L1920) | avaliação determinística completa de uma lista de candidatos | **Sim, é o ancestral direto de `evaluate_strategy()` (3.4.5)** |
| `summarize_validation_tool()` | [tools.py:1699](orchestrator/tools.py#L1699) | só o `validation_summary` | **Sim** — base de `error_summary()` |
| `generate_data_driven_candidates_tool()` | [tools.py:1716](orchestrator/tools.py#L1716) | candidatos sugeridos por heurística | Não |

`SCORE_PRESETS` ([tools.py:32](orchestrator/tools.py#L32)) define 4 perfis de peso do composite score
(`balanced`, `rmse_focus`, `direction_focus`, `robust_smape`) — mantidos pelo campo `score_preset` do
CSV (Seção 4.3).

---

## 4. Formato de entrada de dados (a **preservar**)

### 4.1 Arquivo por modelo

```
./timeseries/mestrado/resultados/<MODELO>/normal/<DATASET>.csv     (sep=";")
```

Colunas: `dataset_index; horizon; regressor; mape; pocid; smape; rmse; msmape; mae; test;
predictions; start_test; final_test`

- Uma linha por (série, janela). `dataset_index` identifica a série; `start_test`/`final_test`
  delimitam a janela.
- `test` e `predictions` são **strings** de arrays (formato `repr` de numpy, podendo conter quebras
  de linha e notação científica). São parseados por regex em
  [context.py:81](orchestrator_langchain/context.py#L81) —
  `re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)`.
- **Ordenação por `start_test` ascendente.** A **última linha** é o teste final (cego); as anteriores
  são as janelas de validação.
- Baselines já combinadas (`mean`, `median`, `dba`, `ADE`, `FFORMA`) usam o **mesmo schema de 13
  colunas**, mas ficam em `resultados/<NOME>/<DATASET>.csv` (sem a subpasta `normal/`).

### 4.2 Estrutura em memória após a ingestão

`generate_all_validations_context()` ([context.py:87](orchestrator_langchain/context.py#L87)) popula
o dicionário global `CONTEXT_MEMORY`:

```python
CONTEXT_MEMORY = {
    "dataset_index": int,
    "models_available": [str, ...],
    "all_validations": {
        "predictions": [ {modelo: [h1..hH], ...},   # janela 0 (mais antiga)
                         {modelo: [h1..hH], ...} ], # janela 1 ...
        "test":        [ [h1..hH],                  # y_true da janela 0
                         [h1..hH] ],
    },
    "predictions": {modelo: [h1..hH]},   # previsões do TESTE FINAL (sem y_true)
    "tools_called": [str, ...],
}
```

`load_validation_from_context()` ([data_contract.py:38](orchestrator/data_contract.py#L38))
normaliza isso para arrays:

- `y_true` — `(n_windows, horizon)`
- `y_preds` — `(n_windows, n_models, horizon)`
- `model_names` — lista, ordem fixada pela janela 0

O horizonte é truncado ao mínimo comum entre `test` e todas as previsões de cada janela.

**Mapeamento para as entradas da Seção 3.1:**

| Entrada da especificação | Onde está hoje |
|---|---|
| `previsoes_validacao` | `CONTEXT_MEMORY["all_validations"]` → `ValidationData` |
| `previsoes_teste` | `CONTEXT_MEMORY["predictions"]` |
| `serie_treino` | **não existe** — ver D3 (reconstruível concatenando `test` das janelas) |

Este formato é **preservado integralmente** na reescrita. A arquitetura ReAct atua depois da Fase 0.

---

## 5. Onde o CSV de resultados é montado e salvo

**Um único lugar, no script de entrada** — nada disso está dentro dos pacotes `orchestrator*`:

| O quê | Local exato |
|---|---|
| Lista de colunas `COLS_SERIE` (52 colunas) | [run_tsf_orchestrator.py:33-92](run_tsf_orchestrator.py#L33-L92) |
| Caminho de saída | [run_tsf_orchestrator.py:170-171](run_tsf_orchestrator.py#L170-L171) — `./timeseries/mestrado/resultados/orchestrator_llm_{version}/<DATASET>.csv` |
| Artefatos LLM (JSON por série) | [run_tsf_orchestrator.py:172](run_tsf_orchestrator.py#L172) — `.../llm_artifacts/<DATASET>/dataset_<i>.json` |
| Criação / migração de schema | [run_tsf_orchestrator.py:181-194](run_tsf_orchestrator.py#L181-L194) (adiciona colunas faltantes e reindexa se o CSV já existir) |
| Extração dos campos LLM do `description` | [run_tsf_orchestrator.py:283-534](run_tsf_orchestrator.py#L283-L534) |
| Cálculo das 6 métricas | [run_tsf_orchestrator.py:555-577](run_tsf_orchestrator.py#L555-L577) |
| **Montagem do `data_serie`** | **[run_tsf_orchestrator.py:579-633](run_tsf_orchestrator.py#L579-L633)** |
| **Gravação (append)** | **[run_tsf_orchestrator.py:635-637](run_tsf_orchestrator.py#L635-L637)** |

O `data_serie` é populado a partir de um único JSON: `result["description"]`, montado em
[orchestrator/pipeline.py:1226-1260](orchestrator/pipeline.py#L1226-L1260), mais `result["debate"]`,
`result["explanations"]`, `result["eval"]` e `result["llm_artifacts"]`. **É esse dicionário
`description` que precisa mudar de forma para alimentar os campos novos da Seção 4.4.**

Observação de compatibilidade: `mape` é calculado com `sklearn.mean_absolute_percentage_error`
(fração), enquanto `pocid` vem de `all_functions.pocid` (percentual 0-100), e `smape`/`msmape`/
`rmse`/`mae` vêm de `all_functions` com assinatura `(forecasts, test_set)` **reshapeados para
`(1, -1)`**. Essa convenção precisa ser mantida byte a byte na reescrita (Seção 4.1).

---

## 6. Utilitários de baixo nível reaproveitáveis vs. lógica de agente a descartar

### 6.1 REAPROVEITAR (utilitários genéricos, sem LLM)

| Função | Local | Uso na nova arquitetura |
|---|---|---|
| `calculate_smape`, `calculate_rmse`, `calculate_msmape`, `calculate_mae`, `pocid` | [all_functions.py:134,1682-1735](all_functions.py#L1682) | Métricas do CSV final — **obrigatório manter idênticas** (4.1) |
| `mape_safe`, `smape_safe`, `rmse_safe`, `pocid_within_sequence`, `compute_metrics_1d`, `MetricConfig` | [orchestrator/metrics.py](orchestrator/metrics.py) | Métricas internas do backtest (tratam zero/NaN) |
| `extract_values`, `read_model_preds` | [context.py:68-85](orchestrator_langchain/context.py#L68-L85) | Fase 0 — ingestão. **Preservar** |
| `ValidationData`, `load_validation_from_context` | [orchestrator/data_contract.py](orchestrator/data_contract.py) | Fase 0 — contrato de dados normalizado |
| `_stl_decompose_components` | [strategies.py:126](orchestrator/strategies.py#L126) | `series_profile()`, `stl_summary()` |
| `_project_simplex` | [strategies.py:27](orchestrator/strategies.py#L27) | Todas as tools de peso (3.4.3) |
| `_trimmed_mean` | [strategies.py:49](orchestrator/strategies.py#L49) | `combine_trimmed_mean()` |
| `_weights_inverse_rmse` | [strategies.py:104](orchestrator/strategies.py#L104) | `weights_inverse_error()` |
| `_weights_exp` / `_weights_poly` | [strategies.py:86-101](orchestrator/strategies.py#L86-L101) | `weights_softmax_neg_error()` |
| `_ridge_weights` | [strategies.py:176](orchestrator/strategies.py#L176) | base de `weights_ols()` |
| `_keep_best_by_ratio` | [strategies.py:69](orchestrator/strategies.py#L69) | `select_top_k()` |
| bloco `method == "dba"` (tslearn `dtw_barycenter_averaging`) | [strategies.py:489](orchestrator/strategies.py#L489), [final_predictor.py:405](orchestrator/final_predictor.py#L405), [combinations/dba.py:29](combinations/dba.py#L29) | `combine_dba()` — **3 implementações duplicadas, unificar em uma** |
| `diebold_mariano` (com correção HLN) | [diagnostics.py:241](orchestrator/diagnostics.py#L241) | `dm_test()` (3.4.1) — **pronto para uso** |
| `rank_stability_kendall` | [diagnostics.py:161](orchestrator/diagnostics.py#L161) | `ranking_stability()` — **pronto para uso** |
| `error_similarity_matrix` | [diagnostics.py:195](orchestrator/diagnostics.py#L195) | `error_correlation()` / `prune_redundant()` |
| `spectral_entropy`, `hurst_rs`, `ljung_box_pvalue`, `heteroscedasticity_ratio`, `drift_signal`, `bias_per_horizon` | [orchestrator/diagnostics.py](orchestrator/diagnostics.py) | Features rápidas de `series_profile()` (substituto parcial de catch22) |
| `paired_bootstrap_score` | [diagnostics.py:312](orchestrator/diagnostics.py#L312) | Guarda-corpo estatístico opcional em `evaluate_strategy()` |
| `_compute_model_aggregates`, `_compute_best_model_per_horizon`, `_compute_disagreement_score` | [tools.py:144-226](orchestrator/tools.py#L144-L226) | `error_summary()` |
| `extract_json_object`, `strip_think_blocks` | [orchestrator/utils.py](orchestrator/utils.py) | Parsing robusto da saída do LLM no laço ReAct |
| `_extract_think_blocks` | [run_tsf_orchestrator.py:95](run_tsf_orchestrator.py#L95) | Extração de `thought` para `react_trajectory_json` |
| `generate_combined_predictions` (avaliação) + `predict_final_from_context` (aplicação) | [strategies.py:211](orchestrator/strategies.py#L211), [final_predictor.py:103](orchestrator/final_predictor.py#L103) | Par backtest/aplicação — o princípio "mesmas funções na Fase 3 e Fase 4" (3.3) **já é a intenção aqui**, mas as duas estão **duplicadas divergentemente** (ver nota abaixo) |
| `metrics.py`, `aux.py` de `combinations/` | [combinations/](combinations/) | Leitura das baselines já calculadas para `baseline_results_json` |

> **Dívida técnica herdada, a resolver na reescrita:** `strategies.py` (backtest) e
> `final_predictor.py` (aplicação final) reimplementam **os mesmos 11 métodos duas vezes**, com
> helpers duplicados (`_project_simplex`, `_weights_inverse_rmse`, `_weights_exp`, `_weights_poly`,
> `_ridge_weights`, `_trimmed_mean`). Elas já divergem: `final_predictor` treina em **todas** as
> janelas, `strategies` respeita o `_train_slice` anti-leakage. O princípio da Fase 4 ("usando
> exatamente as mesmas funções determinísticas usadas no backtest") exige **unificar isso numa única
> implementação parametrizada** — é a maior fonte de risco de inconsistência no código atual.

### 6.2 DESCARTAR (lógica de agente / prompt / debate — só referência histórica)

- Os 5 prompts em `orchestrator_langchain/prompts/` (`pattern_analyst.md`, `proposer.md`,
  `skeptic.md`, `statistician.md`, `orchestrator.md`).
- As 5 factories `create_*_agent` ([agents.py:103-155](orchestrator_langchain/agents.py#L103-L155)) e
  o reexport `orchestrator/agents.py`.
- Toda a orquestração de debate em `orchestrator/pipeline.py`: `run_llm_pipeline` (linhas 642-1305),
  `_validate_actions_against_universe`, `_apply_actions_to_payload`, `_sanitize_candidate_payload`,
  `_resolve_candidate_name`, `ALLOWED_PARAM_EDITS`, `_run_agent_with_retry`.
- As tools que são *interfaces de papel*: `proposer_brief_tool`, `build_debate_packet_tool`,
  `build_fold_cot_context_tool` (a parte de "campeões"/tiers/hints), `_candidate_universe_from_summary`,
  `_suggest_candidates_from_summary`, `resolve_unknown_candidate`, `_recommended_knobs`.
- `orchestrator/run_research_loop.py` (código morto, D8.4).
- A classe `LangchainAgent` ([agents.py:28](orchestrator_langchain/agents.py#L28)) com sua lógica de
  `force_tool_call` + mensagens de "nudge" — o laço ReAct precisa de um loop multi-turno com
  histórico, não de um forçador de tool-call de turno único.

> Conforme a nota do Passo 1, o campo `pattern_analyst_trend_champion` / `_seas_champion` **não** deve
> ser portado a partir desse código. O conceito equivalente (qual modelo melhor acompanha
> tendência/sazonalidade) reaparece dentro de `series_profile_json`, implementado do zero como função
> determinística.

---

## 7. Mapeamento: catálogo da Seção 3.4 → o que já existe

Legenda: ✅ existe e é reaproveitável · 🟡 existe parcialmente, precisa adaptação · ❌ código novo

| Tool (3.4) | Status | Base existente |
|---|---|---|
| `series_profile()` | 🟡 | `_stl_decompose_components`, `spectral_entropy`, `hurst_rs`, `ljung_box_pvalue`. **Falta:** ADF/KPSS (precisa `statsmodels.tsa.stattools`), catch22 (`pycatch22` não instalado), e a própria série (D3) |
| `stl_summary()` | ✅ | `_stl_decompose_components` + lógica de variância explicada de `build_fold_cot_context_tool` |
| `error_summary()` | ✅ | `_compute_model_aggregates` + `_build_validation_summary` |
| `ranking_stability()` | ✅ | `rank_stability_kendall` (falta só a lista de "quem mais mudou de posição") |
| `error_correlation()` | ✅ | `error_similarity_matrix` (retorna par mais redundante; falta agrupamento) |
| `dm_test()` | ✅ | `diebold_mariano` — pronto |
| `select_top_k()` | ✅ | `_keep_best_by_ratio` + agregados por modelo |
| `select_stable()` | 🟡 | derivável de `rank_stability_kendall`, mas por-modelo é novo |
| `prune_redundant()` | 🟡 | `error_similarity_matrix` dá a matriz; o clustering por limiar é novo |
| `weights_inverse_error()` | ✅ | `_weights_inverse_rmse` |
| `weights_softmax_neg_error()` | ✅ | `_weights_exp` (ver ressalva D7) |
| `weights_ols()` | 🟡 | `_ridge_weights` com `l2=0` + `nonneg_simplex=False` |
| `weights_feature_based()` | ❌ | ver D7 — decisão pendente |
| `combine_mean/median/trimmed_mean/weighted/best_single` | ✅ | `generate_combined_predictions` |
| `combine_dba()` | ✅ | 3 implementações; unificar |
| `evaluate_strategy(spec_json)` | 🟡 | `evaluate_strategies_tool` + `evaluate_all`; falta o **histórico ranqueado de tentativas** e a posição no ranking |
| `sanity_check()` | ❌ | novo |
| `list_attempts()` | ❌ | novo — não há conceito de histórico de tentativas hoje |
| **Registry de *handles* de pesos** | ❌ | novo — hoje os pesos são embutidos no `params` do candidato, não referenciados por id |

---

## 8. Formato de saída — checagem contra o Passo 4

O `COLS_SERIE` atual tem **52 colunas**. Após aplicar o Passo 4:

- **13 preservadas intactas** (4.1): `dataset_index, horizon, regressor, mape, pocid, smape, rmse,
  msmape, mae, test, predictions, start_test, final_test`. Todas presentes hoje, mesmos nomes.
- **23 removidas** (4.2): todas as `debate_*`, `approach_*`, `proposer_*`, `skeptic_*`,
  `statistician_*`, `pattern_analyst_*`. Confere com o CSV atual — 23 colunas exatas.
- **14 mantidas/ressignificadas** (4.3): `description, selection_explanation, when_good,
  decision_report, llm_artifacts_path, score_preset, tool_missing, tools_called, best_strategy_name,
  best_strategy_method, best_strategy_params, predict_debug, selected_base_models,
  weights_by_horizon`. Todas presentes hoje.
- **2 órfãs não classificadas pela especificação:** `final_candidate_names` e
  `final_candidate_count` ([run_tsf_orchestrator.py:64-65](run_tsf_orchestrator.py#L64-L65)). Não
  aparecem em 4.1, 4.2 nem 4.3. Semanticamente pertencem ao mecanismo antigo (são o ranking pós-debate,
  preenchidas em [run_tsf_orchestrator.py:522-534](run_tsf_orchestrator.py#L522-L534) a partir de
  `result["eval"]["ranking"]`). **Proposta: remover junto com 4.2**, já que o equivalente novo é
  `react_trajectory_json` + `list_attempts()`. Confirmar antes de implementar.
- **18 novas** (4.4). Nenhuma existe hoje.

Total previsto: 13 + 14 + 18 = **45 colunas** (se as 2 órfãs forem removidas).

Sobre a convergência sugerida em 4.3 — **proposta**: manter `description` como o JSON estruturado
completo (equivalente ao `decision.json` da Seção 3.1, é o que ele já é hoje), manter
`decision_report` como o resumo de uma linha legível, e **colapsar `selection_explanation` +
`when_good` em `justificativa_final`**, já que os três descreveriam o mesmo texto causal na nova
arquitetura. Decisão a confirmar.

---

## 9. Ambiente e dependências

O ambiente usado é o conda `agno` (`conda activate agno`). O código atual **já depende** de pacotes
que não estão em `agno.yml` (o export está desatualizado): `statsmodels` (STL, Ljung-Box),
`tslearn` (DBA), `langchain-core`, `langchain-ollama`. A arquitetura nova acrescenta pelo menos
`pycatch22` (features catch22) e, se o laço ReAct for construído como grafo, `langgraph`.

A lista consolidada de instalações extras está em **[EXTRA_DEPENDENCIES.txt](EXTRA_DEPENDENCIES.txt)**.
O repositório também foi preparado para `uv` (`pyproject.toml` + entradas no `.gitignore`), como
alternativa ao conda — nada foi instalado nem executado.

O LLM é servido por **Ollama local em `http://127.0.0.1:11501`**, hardcoded em
[orchestrator_langchain/agents.py:42](orchestrator_langchain/agents.py#L42). A Seção 3.5 exige troca
de modelo por papel via config/env sem alterar código — hoje o `base_url` e o `temperature` por papel
estão fixos no código (`ModelConfig.temperature` é aceito mas
[ignorado](orchestrator_langchain/agents.py#L103-L144): as factories usam `0.3`/`0.2`/`0.15`
hardcoded, não o valor passado). Mais um ponto a corrigir na reescrita.

---

## 10. Decisões tomadas e pendências

### Decidido

| # | Questão | Decisão |
|---|---|---|
| 1 | Branch base | **Continuar em `v1_proposta`.** `halmoc` fica de lado. |
| 2 | `serie_treino` | **Carregar do arquivo-fonte original**, em `../forecasting_datasets` (fora do repo). Novo parâmetro em `exec_dataset_orchestrator` com o nome do arquivo. Mapeamento validado em D3, com guarda-corpo em runtime. |
| 3 | `train_window` | **Mudar para 4** → 3 janelas de validação reais, alinhando o orquestrador às baselines já calculadas. |
| 4 | Campos de texto do CSV | **Colapsar `selection_explanation` + `when_good` em `justificativa_final`.** `description` continua sendo o JSON estruturado (o `decision.json` da Seção 3.1) e `decision_report` o resumo de uma linha. |
| 5 | Pool de modelos | **Manter os 19 ativos** (não descomentar as variantes `ridge`). |
| 6 | Bug skeptic↔statistician | **Não corrigir** — os papéis são descartados pela arquitetura nova. Fica registrado em D8.1 para a leitura dos CSVs v1 já gerados. |

### Assinatura proposta para a Fase 0

```python
exec_dataset_orchestrator(
    models,
    dataset,                       # "ETTH1"  -> nome da pasta de resultados
    source_file,                   # NOVO: "ETTh1.csv" | "nn5_weekly_dataset.tsf" | ...
    source_dir="../forecasting_datasets",   # NOVO, configurável
    ...
)
```

O loader lê `.tsf` (parser Monash), aplica o pré-processamento registrado por dataset (filtro do ANP,
frequência `half_hourly → 30min` do ETTm) e **valida cada série carregada** contra a coluna `test` do
CSV de resultados antes de devolvê-la.

### Pendências

1. **`weights_feature_based`:** treinar o meta-modelo FFORMA uma vez por dataset na Fase 0, ou usar o
   fallback `softmax(-SMAPE)` por série? — ver D7.
2. **`final_candidate_names` / `final_candidate_count`:** a especificação não os classifica em 4.1,
   4.2 nem 4.3. Proposta: remover junto com os campos de debate (o equivalente novo é
   `react_trajectory_json` + `list_attempts()`).
3. **`base/mes_11_venda_mensal.tsf` está dentro do repo**, enquanto todos os outros fontes ficam em
   `../forecasting_datasets`. Copiar para lá (uniformiza o `source_dir`) ou tratar o ANP como caso
   especial com caminho relativo à raiz?
