# Estado atual do repositório versus v1_proposta

Este documento descreve, de forma consolidada, o que o repositório implementa hoje,
o que foi feito desde a escrita inicial da proposta (branch `v1_proposta`) e o que
precisa ser removido, adicionado ou reescrito em `overleaf/proposta.tex` e
`overleaf/resultados.tex`. Ele foi montado a partir de leitura direta do código,
do diff `v1_proposta..v1_proposta_final`, da bateria `aprovacao_solida/` e dos
resultados dos runs em `timeseries/mestrado/resultados/`.

Fontes principais: `overleaf/proposta.tex` (método da proposta original),
`overleaf/resultados.tex` (capítulo de resultados já escrito com a rodada v5),
`aprovacao_solida/DOCUMENTO_ADICOES.md` (guia antigo de edições),
`melhorias/PLANO_IMPLEMENTACOES.md`, `COMO_FUNCIONA_V5.md`, `METODO.md`,
`ENTENDENDO_A_PROPOSTA.md` e os commits listados na Seção 5.

---

## 1. Mapa do repositório

Ativo, núcleo do método:

- `orchestrator_react/`: implementação atual do CREST em Python. É o pacote que
  roda nos experimentos. Módulos: `config.py` (todos os parâmetros e presets),
  `pipeline.py` (Fases 0 a 5), `state.py` (histórico, backtest, score, estratégias
  finais), `pool.py` (sementes e pools estáveis), `weighting.py` (receitas de pesos),
  `selection.py` (seleção de modelos), `tools.py` (catálogo de ações),
  `registry.py` (catálogo fechado e validação), `prompts.py` (prompts v1 e v2),
  `react_loop.py` (ciclo ReAct), `llm.py` (cliente Ollama), `meta_model.py`
  (meta-modelo entre séries), `ingest.py` e `data_source.py` (dados), `metrics.py`
  e `combiners.py`, `csv_writer.py` (CSV e telemetria), `features.py`,
  `phases.py`, `insights_trabalhos.md`, `ARQUITETURA.md`.
- `run_tsf_orchestrator.py`: executa o CREST em um dataset (fases, LLM, CSV).
- `run_tsf_batch.py`: executa vários datasets e vários combinadores (LLM) em
  sequência, um log por combinação.
- `run_tsf_baselines.py` e `run_tsf_batch_baseline.py`: braço determinístico,
  sem LLM (sementes aplicadas direto), corresponde ao `sem_agente`.
- `run_tsf_experiments.py`, `run_tsf_regressors.py`, `experiments_final.py`,
  `regressors_experiments.py`, `m4_dataset.py`, `deseasonal.py`,
  `all_functions.py`: scripts de experimentos anteriores (modelos individuais,
  datasets), ainda usados para gerar os resultados dos 19 preditores.
- `aprovacao_solida/`: pacote de validação e aprovação com a bateria de
  experimentos determinísticos (Seção 6), com `run_all.py` e `DOCUMENTO_ADICOES.md`.
- `melhorias/PLANO_IMPLEMENTACOES.md`: plano das melhorias implementadas e das
  descartadas.
- `tests/`: suíte de testes do `orchestrator_react` (14 arquivos, 515 testes).
- `timeseries/`: dados e resultados. `timeseries/mestrado/resultados/<método ou
  run>/<DATASET>.csv` contém os resultados por série de cada preditor e de cada
  run do orquestrador. `timeseries/mestrado/dados` (fontes .tsf) e artefatos.
- `outputs/resultados/`: resultados derivados (bateria, ablação, tabelas, figuras).
- `overleaf/`: textos da dissertação. Hoje fora do git (untracked): `proposta.tex`,
  `resultados.tex`, `background.tex`, `revisao.tex` e figuras.
- `logs/`: logs de execução. Voltaram a ser commitados no commit `f3b87b101`.
- `base/`, `combinations/` (FFORMA e ADE de referência), `orchestrator/`
  (módulo `diagnostics.py` é usado pelo `state.py` para Diebold-Mariano e
  bootstrap; o restante é legado), `orchestrator_langchain/` (protótipo antigo),
  `multimodal/`, `image_experiments.py`, notebooks e `.drawio` (legado ou apoio).

Legado que não participa dos resultados atuais: `orchestrator_langchain/`,
`multimodal/`, `experiments_final.py` e afins, `violin/`, `diagnostics/logs`.

---

## 2. Ambiente e execução

- Ambiente: `conda activate agno` (Python 3.11), Ollama em `http://127.0.0.1:11434`.
- Rodar um dataset:

```bash
python3 run_tsf_orchestrator.py --dataset ANP_MONTHLY --source mes_11_venda_mensal.tsf \
    --combinator gpt-oss:20b --version v2_medium \
    --reasoning medium --catalog-v2 --prompt-v2
```

- Rodar vários datasets e modelos:

```bash
python3 run_tsf_batch.py \
    --datasets ETTM1 ETTM2 ETTH1 ETTH2 ANP_MONTHLY NN5_WEEKLY_DATASET M4_WEEKLY_DATASET \
    --combinators gpt-oss:20b qwen3.8:27b muse-glimmer:30b \
    --version v2_medium \
    -- --reasoning medium --catalog-v2 --prompt-v2
```

- Opções relevantes hoje: `--final-strategy {argmin,ensemble,seed_gate,mcs}`,
  `--mcs-alpha`, `--mcs-score-margin`, `--mcs-max-members`, `--reasoning
  {off,on,low,medium,high,xhigh,max}`, `--catalog-v2`, `--prompt-v2`,
  `--no-extra-actions`, `--score-preset` (`balanced`, `rmse_focus`,
  `direction_focus`, `robust_smape`, `scale_free_safe`, `auto_loo`),
  `--windows`, `--max-iterations`, `--early-stop-patience`, `--limit`.
- Testes: `python3 -m pytest tests/ -q` (515 testes na última execução).
- Saídas: CSV por dataset em `timeseries/mestrado/resultados/orchestrator_react_<version>/`,
  `agent_summary.json` na mesma pasta (o arquivo é sobrescrito a cada dataset,
  então só guarda o último), e logs em `logs/`.

---

## 3. O método implementado hoje, ponto a ponto

### 3.1 Protocolo e pontuação

Igual ao texto da proposta:

- 3 janelas de validação e 1 janela de teste posterior; horizonte comum.
- Seleção de pool com leave-one-window-out; ajuste de pesos com janela expansiva
  (`nested_selection=True`, `backtest_mode="expanding"`).
- Score composto normalizado pela média simples do pool (`s0`), com pesos
  `balanced` = RMSE 0,3, SMAPE 0,3, MAPE 0,2, POCID 0,2, exatamente a
  Equação `eq:score` da proposta (`orchestrator_react/config.py:19`).
- POCID calculado por janela e depois médio; as demais métricas sobre as janelas
  concatenadas.

Novo desde a proposta:

- Presets alternativos de pesos e `auto_loo`, que escolhe os pesos por
  leave-one-window-out por série (`state.auto_score_weights`). Implementado e
  testado, mas não usado no run final.
- Análise de sensibilidade dos presets na bateria (`06_sensibilidade_score`):
  a seleção muda em 41 a 51 por cento das séries, mas o SMAPE de teste varia
  pouco, o que sustenta a justificativa a priori dos pesos.

### 3.2 Meta-modelo entre séries

Igual à proposta: 26 características (4 interpretáveis mais 22 do catch22),
um único modelo multi-saída transformado em pesos por softmax, objetivo FFORMA,
treinado com leave-one-series-out, uma vez por dataset
(`orchestrator_react/meta_model.py`). A semente correspondente entra no piso
quando o modelo é treinado (datasets com 20 ou mais séries).

### 3.3 Prior de estratégias

Igual à proposta: média, entre as demais séries, do score normalizado de cada
estratégia fixa (`pipeline._build_strategy_priors`). Hoje o prior aparece no
cartão do dataset mostrado ao agente e, opcionalmente, alimenta a estratégia
final `prior_blend` (opt-in).

### 3.4 Sementes

Igual à proposta: média, mediana e DBA sobre o pool completo; média e média
aparada sobre os `kappa` preditores mais estáveis, para `kappa` em {5, 7, 9}
(`pool.SEED_STABLE_POOLS`); e, quando disponível, a ponderação do meta-modelo.
Total de 10 estratégias semeadas antes de o agente atuar; o piso do run é a
melhor delas por validação. A hipótese de incluir a mediana nas pools estáveis
foi testada na bateria (`05_sementes_medianas`) e não foi adotada: melhora o M4,
piora ANP e NN5.

### 3.5 Diagnóstico por série

Igual à proposta: perfil (26 características), avaliação do conjunto
(RMSE por janela, estabilidade de ranking por Kendall, correlação de resíduos) e
diagnóstico determinístico (regime, previsibilidade, indicação, riscos, narrativa),
com opção de leitura por LLM restrita ao vocabulário fechado. Novo: no catálogo v2
os seis diagnósticos saíram do catálogo de ações e passaram a integrar o contexto
do turno; no prompt v2 o diagnóstico alimenta o bloco LEADER DIAGNOSTICS.

### 3.6 Catálogo de ações: o que mudou de forma mais visível

A proposta descreve 24 ações em cinco categorias (6 de diagnóstico, 3 de seleção
de pool, 6 de pesos, 6 de combinação, 3 de validação). O catálogo v2
(`--catalog-v2`) reorganiza isso para 9 ferramentas mais o aceite (10 entradas
nominais), ou 12 ferramentas mais o aceite (13 entradas) quando as ações extras
estão ligadas, que é o padrão atual. Com 3 janelas, `weights_ols` fica retida
(exige 5 janelas), então o agente vê uma ferramenta a menos.

Mapeamento proposta para o catálogo v2:

- Diagnóstico: as 6 ações saem do catálogo; o conteúdo já vai no contexto do turno.
- Seleção de pool: `select_top_k`, `select_stable` e `prune_redundant` permanecem;
  `select_forward` (seleção gulosa de Caruana) é nova.
- Pesos: `weights_inverse_error` e `weights_error_trend` permanecem;
  `weights_softmax_neg_error` sai como duplicata conceitual de `inverse_error`;
  `weights_feature_based` e `weights_pooled_meta_model` viram uma única
  `weights_meta_model(scope="series" ou "dataset")`; `weights_ols` permanece no
  registro mas fica retida com 3 janelas.
- Combinação: as 6 ações `combine_*` saem do catálogo como açúcar sintático; a
  mesma capacidade é acessada por `evaluate_strategy`, que monta e pontua a
  estratégia a partir de `combine`, `pool`, `weights` e `model`.
- Validação: `sanity_check` e `list_attempts` saem (o conteúdo já está no
  contexto); `evaluate_strategy` permanece como a única ação que insere tentativa
  no histórico.
- Novas ações opcionais (`extra_actions=True`, padrão): `weights_best_per_horizon`,
  `weights_topk_per_horizon` e `weights_forward` (Caruana como pesos). Testadas
  offline em `07_novas_combinacoes` e reprovadas como escolha forçada; ficam
  expostas como diversificação para o agente escolher.
- Interface tolerante em `evaluate_strategy`: aceita o nome do método de pesos
  (`weights="inverse_error"`), reajusta handle de outra pool para a pool pedida e
  aceita `combine="weights_inverse_error"`. Foi uma correção medida: a rodada v2
  antiga tinha 270 falhas de argumento; com a correção, o run final tem 9.

### 3.7 Prompt

- v1 (`build_system_prompt`, ainda disponível): prosa densa, catálogo completo,
  formato de saída `text` por padrão.
- v2 (`build_system_prompt_v2`, usado nos runs finais): procedimento em quatro
  passos, catálogo agrupado por fase com gatilho de uso em cada descrição, dois
  exemplos trabalhados de tipos diferentes de estratégia, bloco LEADER DIAGNOSTICS
  a cada turno e regra explícita de parada (aceitar semente quando o veredito é
  indistinguível, aceitar estratégia própria separada, mudar de pool ou de tipo
  quando os pesos voltam quase uniformes). Formato `xml` por padrão.
- A proposta atual não descreve nada disso; o prompt v2 é material novo.

### 3.8 Ciclo, parada e decisão final

- ReAct com `Thought`, `Action`, `Action Input`; validação de formato; erros
  estruturados devolvidos ao agente; resposta vazia ou erro de conexão reexecuta
  até um limite, sem consumir iteração.
- Parada: aceite explícito, estagnação (4 propostas de avaliação consecutivas sem
  melhora de 0,01 por cento relativo) ou orçamento de 12 iterações. Só chamadas de
  `evaluate_strategy` contam para a paciência.
- Estratégia aplicada: por padrão o argmin do histórico (incluindo as sementes),
  com o aceite registrado e a divergência marcada quando o agente aceita algo que
  não é o líder. Isso é exatamente o texto da proposta.
- Estratégias finais alternativas, implementadas e medidas, não usadas no run
  final: `ensemble` (média softmax das top-m), `seed_gate` (volta para a semente
  salvo margem), `prior_blend` (encolhe o score para o prior do dataset) e `mcs`
  (comitê do conjunto que o Diebold-Mariano não separa do líder, com guardas de
  margem de score e de escala de previsão). No run final com `mcs`, o resultado
  ficou levemente pior que o argmin e estatisticamente empatado com tudo.

### 3.9 Reasoning e modelos

- O controle `--reasoning` envia o campo `think` do Ollama. Booleanos são
  universais; níveis nomeados são por modelo: gpt-oss `low/medium/high`,
  qwen3.8 `low/medium/xhigh`, muse-glimmer `low/medium/high/max`, gemma4 apenas
  `on/off`.
- O run final dos três modelos usa reasoning `medium` (único nível comum, junto
  com `low`).
- `resultados.tex` hoje compara gpt-oss:20b, qwen3:30b-a3b e gemma4:26b (rodada
  v5). O desenho final compara gpt-oss:20b, qwen3.8:27b e muse-glimmer:30b
  (muse ainda incompleto). gemma4 não tem CSVs no repo atual.

### 3.10 Telemetria

Nova desde a proposta: por série, `agent_trace` (turnos, tempo por etapa, líder
antes e depois, decisão, motivo de parada, métricas derivadas) e 13 colunas
`agent_*` no CSV; por dataset, `agent_summary.json`. A subseção de procedência
da proposta está comentada no `.tex`; a telemetria é o material que a substitui.

### 3.11 Testes

`tests/` cobre catálogo v2, telemetria, auto_loo, novas receitas, garantias,
nested selection, meta-modelo, pipeline, react loop, csv writer. 515 testes
passando no estado atual.

---

## 4. O que mudou desde a v1_proposta (código)

Commits desde `v1_proposta` até hoje (16):

- `f99d2f91d`: bateria determinística, seed_gate, thinking salvo.
- `382408c4e`: telemetria completa, pesos `auto_loo` antes do loop, novas
  combinações (`best_per_horizon`, `topk_mean_per_horizon`, `forward`).
- `61ab736b8`: catálogo v2 (10 entradas), prompt v2 com exemplos e diagnóstico
  do líder, `select_forward`, merge dos meta-modelos.
- `4844de300` e `9c052187f`: refinamentos do prompt v2 e descrições no estilo
  "Use when"; `extra_actions` ligado por padrão.
- `7a5cff0c2`: resultados da bateria de aprovação.
- `410485381`: salvamento incremental do offline e dump de score por janela.
- `d1b4d52ab`: remoção de logs antigos do git (voltaram depois).
- `ca906447e`: reasoning por modelo, mapeamento on/off e níveis nomeados.
- `e77751c54`: rodada v2 batch low.
- `229f3d924`: interface tolerante do `evaluate_strategy`.
- `b2185c722` e `41ade270f`: estratégia final `mcs` e sua versão robusta.
- `f3b87b101`, `277d37077`: rodada final e logs.

Arquivos de núcleo alterados: `config.py`, `csv_writer.py`, `llm.py`,
`pipeline.py`, `prompts.py`, `react_loop.py`, `registry.py`, `selection.py`,
`state.py`, `tools.py`, `weighting.py`, `run_tsf_orchestrator.py`,
`run_tsf_baselines.py`; novos testes `test_auto_score_weights.py`,
`test_catalog_v2.py`, `test_new_recipes.py`, `test_telemetry.py`.

Novos artefatos de experimento: `aprovacao_solida/` inteiro (Seção 6),
`melhorias/PLANO_IMPLEMENTACOES.md`, `COMO_FUNCIONA_V5.md`,
`ENTENDENDO_A_PROPOSTA.md`, `ENTENDENDO_O_AGENTE.md`,
`ENTENDENDO_AS_METRICAS.md`, `METODO.md`, `PLANO_AGENT_TOOLS.md`,
`PLANO_EXECUCAO_V5.md`, `RELATORIO_TECNICO_COMBINACAO.md`, `EXPLORACAO.md`,
`PARECER_BANCA_CREST.md` (interno, não commitado) e
`aprovacao_solida/09_catalogo_deterministico/` (script pronto, ainda não
commitado, para enumeração determinística do catálogo).

---

## 5. Experimentos e resultados disponíveis

### 5.1 Runs do orquestrador

- v5 (gpt-oss:20b, configuração original de 24 ações, reasoning padrão):
  `orchestrator_react_v5`. É a base do `resultados.tex` atual.
- v5 com qwen3:30b-a3b: `orchestrator_react_v5_qwen`.
- v5 repetido: `orchestrator_react_v5_x2`. v5 com gemma4:26b
  (`orchestrator_react_v5_gemma26`) está sem CSVs no repo.
- Braço sem agente: `orchestrator_baseline_v1` (piso aplicado por série).
- v2 low (argmin, sem correção de interface): `orchestrator_react_v2_low`.
- v2 final (low mais MCS): `orchestrator_react_v2_final`.
- v2 medium (argmin, com correção): `orchestrator_react_v2_medium`.
- Final comparável: `orchestrator_react_v2_medium_x1_gpt-oss-20b`,
  `..._qwen3.8-27b` (completos) e `..._muse-glimmer-30b` (ETTs completos; ANP
  vazio; NN5 e M4 ainda no servidor).

### 5.2 Resultado central (todos os números em SMAPE médio por série)

Total ponderado nas 680 séries:

| Braço | SMAPE | POCID |
|---|---|---|
| Piso determinístico (sem_agente) | 0,1369 | 58,43 |
| v2_medium (gpt-oss, argmin) | 0,1370 | 58,26 |
| v5 (gpt-oss, catálogo original) | 0,1376 | 57,86 |
| v2_medium_x1 qwen3.8 | 0,1376 | 58,24 |
| v5 qwen3:30b-a3b | 0,1378 | 57,51 |

Por dataset, no run final comparável:

| Dataset | Piso | gpt-oss medium | qwen3.8 medium | muse (ETTs) |
|---|---|---|---|---|
| ANP | 0,2197 | 0,2197 | 0,2188 | — |
| NN5 | 0,1157 | 0,1157 | 0,1184 (p=0,041 pior) | — |
| M4 | 0,0914 | 0,0915 | 0,0920 | — |
| ETTH1 | 0,2262 | 0,2272 | 0,2475 | 0,2290 |
| ETTH2 | 0,1799 | 0,1799 | 0,1762 | 0,1799 |
| ETTM1 | 0,4604 | 0,4598 | 0,4664 | 0,4654 |
| ETTM2 | 0,2030 | 0,2030 | 0,1895 | 0,2030 |

Processo: gpt-oss medium usa 2,9 iterações e 0,5 avaliação de estratégia por
série; qwen3.8, 1,9 e 0,45; muse nos ETTs, 6,9 e 1,8, com 87 segundos de LLM por
série. Falhas de ferramenta no run final: 9 (gpt), 9 (qwen), 36 (muse nos ETTs).
Nos 28 ETTs, os três modelos ficam no nível do piso (0,2674 do piso contra
0,2675 gpt, 0,2693 muse, 0,2699 qwen).

Leitura: nenhum agente ou modelo supera o piso de forma significativa; a única
diferença significativa por dataset é o qwen3.8 pior na NN5. O agente, com mais
raciocínio, converge para não mexer.

### 5.3 Bateria determinística (`aprovacao_solida/`, resultados em `outputs/resultados/aprovacao_solida/`)

- 01 busca sem LLM sobre o mesmo catálogo e orçamento: nenhuma busca bate o piso;
  a única diferença significativa após Holm é a busca aleatória pior no POCID do
  M4.
- 02 replay vencedor contra melhor semente: a semente ganha no pooled (0,1369
  contra 0,1377) e em ANP, NN5, M4 e ETTH1; gates por margem e por veredito não
  recuperam a diferença.
- 03 estatística por dataset (Holm) e 04 latência e custo (artefatos v5 são do
  esquema antigo; só horas totais: ANP 6,95, NN5 2,07, M4 5,84 no gpt-oss; qwen
  no M4, 15,26).
- 05 medianas nas sementes: melhora o M4 (0,0914 para 0,0886), piora ANP e NN5 em
  pouco; não adotada.
- 06 sensibilidade dos pesos: pouca variação de teste entre presets.
- 07 novas combinações: nenhuma passa no critério de adoção; `best_single` pela
  validação é o retrato do winner's curse.
- 08 reasoning e harness: insumos para o controle de reasoning.
- Ablação oficial (v5, `outputs/resultados/ablacao/`): agente e sem agente
  indistinguíveis (SMAPE p=0,1403), 54,9 por cento de estratégias idênticas.

### 5.4 Estudos exploratórios (fora da dissertação, para resposta a banca)

- Consistência entre janelas (3 e 24 janelas): nenhuma regra bate o piso; a
  penalidade por incerteza (LCB) é a menos ruim e converge para não mexer.
- Meta-seleção por leave-one-window-out: o pseudo-teste correlaciona 0,83 com o
  teste real no nível de regra, mas a escolha da regra piora o resultado.
- Meta-modelo por candidato (GBM) com LODO por dataset e por série: perde do piso.
- Prior cross-série com encolhimento: perde do piso em todos os cenários testados.
- Média das top-m por validação (ensaio do comitê): melhora o argmin em todos os
  datasets e bate o piso nos ETTs, mas perde no pooled.
- Número de janelas: só o ANP tem mais de 4 origens (31); NN5, M4 e ETTs têm
  exatamente 3 de validação e 1 de teste, então o estudo de mais janelas só é
  possível no ANP, e nele mais janelas não consertam a seleção.
- Enumeração determinística do catálogo v2: script pronto em
  `aprovacao_solida/09_catalogo_deterministico/run_catalog_enum.py`, testado em
  uma série, ainda não rodado em escala.

---

## 6. Diferenças a aplicar nos documentos

### 6.1 `overleaf/proposta.tex`

Decisão tomada: o capítulo de método deve descrever a configuração efetivamente
usada nos resultados reportados, que é a configuração final: catálogo v2, prompt
v2, reasoning medium e decisão final por argmin. Manter as 24 ações da proposta
enquanto os resultados principais vêm dos runs finais seria inconsistente, e a
banca pode cobrar exatamente isso. A rodada v5 (catálogo original de 24 ações e
prompt v1) passa a ser a ablação de catálogo, com o resultado medido de que a
simplificação não altera a acurácia. Detalhe útil: o v5 rodou com o reasoning
padrão do gpt-oss, que já é medium, então a comparação v5 contra v2_medium isola
catálogo e prompt, não o esforço de raciocínio.

Itens a alterar:

- Considerações iniciais: a frase "com o objetivo de superar os resultados"
  precisa ser reenquadrada para investigar se e em que condições o agente supera;
  o resultado negativo é parte do desenho.
- Visão geral e pseudocódigo: atualizar a contagem de ações (catálogo v2: 10
  entradas nominais, 13 com as três extras, e `weights_ols` retida com 3 janelas,
  então o agente vê 11 ferramentas mais o aceite) e a descrição do catálogo; a
  figura da arquitetura precisa refletir o pipeline real (Fases 0 a 5 com
  telemetria e estratégia final).
- Mecanismo de pontuação: manter a equação; acrescentar a nota sobre os presets,
  o `auto_loo` e a análise de sensibilidade.
- Meta-modelo e prior: manter; mencionar que o prior alimenta o cartão do dataset
  e a estratégia `prior_blend` como ablação.
- Caracterização e diagnóstico: manter; mencionar que os diagnósticos passaram a
  ir no contexto do turno e que o prompt v2 usa o bloco LEADER DIAGNOSTICS.
- Catálogo: substituir as cinco tabelas de 24 ações por uma tabela do catálogo v2
  e um parágrafo de mapeamento (o que saiu para o contexto, o que foi agrupado em
  `evaluate_strategy` e em `weights_meta_model`, e o que é novo: `select_forward`
  e as três ações extras), deixando para o capítulo de resultados a ablação que
  mostra que a simplificação não perde acurácia.
- Ciclo de decisão: manter ReAct; acrescentar o prompt v2 (procedimento,
  exemplos, bloco LEADER DIAGNOSTICS, regra de parada) e a telemetria como
  mecanismo de rastreabilidade, no lugar da subseção de procedência comentada.
- Critérios de encerramento: manter os três; detalhar que a paciência conta
  apenas propostas de avaliação.
- Aplicação: manter o argmin; acrescentar um parágrafo sobre as estratégias finais
  alternativas implementadas e testadas (ensemble, seed_gate, prior_blend, mcs) e
  o motivo de manter o argmin.
- Considerações finais: atualizar o fecho com a ablação de catálogo e o resultado
  negativo.

### 6.2 `overleaf/resultados.tex`

O capítulo já está escrito com a rodada v5 e a ablação. O que precisa mudar:

- Seção "Escolha do modelo de linguagem": hoje compara gpt-oss, qwen3:30b-a3b e
  gemma4:26b. Ou trazer os CSVs do gemma4 e manter, ou substituir por qwen3.8:27b
  e muse-glimmer:30b (temos números completos de gpt-oss e qwen3.8 e ETTs do
  muse). A tabela de custo precisa ser refeita com os números novos.
- Seções por dataset: os números são do v5. Se a opção A for adotada, manter e
  acrescentar uma subseção de replicação com o v2_medium (mesmos datasets,
  diferenças dentro do ruído). Se a opção B for adotada, substituir as tabelas.
- Análise do comportamento do agente: atualizar frequência do catálogo e motivo de
  encerramento com o catálogo v2 (as ações de diagnóstico e os `combine_*` não
  aparecem mais; aparecem `select_forward` e as extras) e com a telemetria nova
  (aceite em 95 por cento no v2, 82 por cento de líderes semente, 62 por cento do
  tempo após a última melhoria).
- Acrescentar as subseções novas do guia antigo (busca sem LLM e vencedor versus
  semente) com os números da bateria completa dos 7 datasets, e remover a
  restrição "na ANP" que está no guia.
- Acrescentar o estudo de ablação do catálogo v2, da interface e do reasoning
  (v2_low, v2_final, v2_medium), e a comparação final entre modelos.
- Cuidado com números desatualizados: a escolha do modelo já cita "o mesmo padrão
  já observado na comparação entre gpt-oss e Qwen3", texto que precisa refletir a
  configuração final.

### 6.3 `overleaf/background.tex` e `overleaf/revisao.tex`

- Conferir se a revisão cobre Model Confidence Set, winner's curse e seleção de
  modelos com validação curta (usados nas discussões novas), e se os LLMs citados
  incluem gpt-oss, qwen3.8 e muse-glimmer (ou uma nota de que a família Qwen foi
  atualizada).
- Conferir a descrição do protocolo de janelas e da métrica POCID, que continuam
  válidos.

### 6.4 `aprovacao_solida/DOCUMENTO_ADICOES.md`

Está desatualizado: foi escrito quando só existiam resultados da ANP, a bateria
não tinha rodado, e as seções 4 ("o que falta rodar") e 6 (sensibilidade) já
foram executadas. Precisa ser reescrito com os números de todos os datasets, com
as decisões finais (argmin, reasoning medium) e com a separação entre o que entra
na dissertação e o que fica como resposta a banca (Seção 5.4 deste documento).

---

## 7. Pendências e decisões abertas

1. Muse: faltam NN5 e M4 (rodando no servidor). Os ETTs já mostram que ele não
   muda a história; completar a tabela quando chegar.
2. gemma4: decidir entre trazer os resultados da rodada antiga ou removê-lo da
   comparação de modelos, substituindo por qwen3.8 e muse. Ele não aceita níveis
   de reasoning (só on/off), o que complica a comparação com medium.
3. Decisão editorial registrada na Seção 6.1: o método descrito passa a ser a
   configuração final (catálogo v2, prompt v2, reasoning medium, argmin), e a
   rodada v5 vira a ablação de catálogo. Confirmar com o orientador, se ele
   precisar ser avisado da mudança em relação ao texto da proposta.
4. MCS: manter no código como ablação e reportar o resultado negativo (reduz o
   dano do argmin, não bate o piso no pooled).
5. Logs: o git voltou a receber logs de execução no commit `f3b87b101`. Decidir se
   ficam versionados ou se saem do repo (são muitos MB).
6. `aprovacao_solida/09_catalogo_deterministico/` e `PARECER_BANCA_CREST.md` estão
   untracked; decidir se entram no repo (o parecer contém menções ao bug do DBA
   que não devem ir para a dissertação).
7. `overleaf/` está fora do git; decidir se versiona ou mantém separado.

---

## 8. Referência rápida de números para citar

- Protocolo: 7 datasets, 680 séries, 19 preditores, 3 janelas de validação,
  1 de teste, horizonte por dataset.
- Piso determinístico (melhor semente por validação): SMAPE 0,1369, POCID 58,43.
- v5 (catálogo 24): SMAPE 0,1376, POCID 57,86.
- v2_medium (catálogo enxuto, reasoning medium, argmin): SMAPE 0,1370,
  POCID 58,26, 632 de 680 séries empatadas com o piso.
- qwen3.8 medium: SMAPE 0,1376; única diferença significativa por dataset é pior
  na NN5 (p_holm=0,041).
- Ablação v5: agente contra sem agente, SMAPE p=0,1403; 54,9 por cento de
  estratégias idênticas.
- Winner's curse: nos runs v5 e v2, quando o agente lidera a validação, o teste
  melhora em cerca de 46 por cento e piora em 54 por cento das séries.
- Decisão final: o piso determinístico é a política de melhor média; o agente é
  mecanismo de exploração, justificativa e auditoria, não fonte de ganho de
  acurácia mensurável.
