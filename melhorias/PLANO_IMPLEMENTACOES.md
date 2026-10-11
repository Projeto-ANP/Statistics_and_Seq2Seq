# Plano de implementações — ler este arquivo antes de implementar

Este arquivo é o registro das ideias de melhoria para o CREST, **separadas por
etapas**, para serem implementadas quando o usuário pedir ("implemente a etapa
X"). Ele deve ser suficiente por si só: quem for implementar não precisa
reler a conversa, só este arquivo.

## 0. Convenções (obrigatórias)

1. **Uma pasta por alteração**: `melhorias/01_forward_selection/`,
   `melhorias/02_por_horizonte/`, etc., com código, `README.md` e
   `ALTERACOES_NO_TEXTO.md`.
2. **Sem `.sh` executável**: os comandos no README devem ser `python3 ...`
   copiáveis direto (loops inline com `bash -c '...'` quando necessário, sem
   `chmod`).
3. **Teste determinístico offline antes de qualquer LLM**: nenhuma melhoria vai
   para a rodada do agente sem antes passar pelo protocolo determinístico
   (mesmo catalogo/orçamento, sem Ollama). Se não passar, vira resultado
   negativo documentado — que também tem valor.
4. **Opt-in, default não muda**: tudo atrás de flag (ex.: `config.forward_selection
   = False`). Runs antigos continuam reproduzíveis.
5. **Regra de adoção**: só adotar se (a) melhorar de forma consistente na
   validação E (b) confirmar na janela de teste. Jamais escolher pelo teste.
6. **Documento**: se a alteração for adotada, escrever em
   `ALTERACOES_NO_TEXTO.md` exatamente o que muda em
   `overleaf/proposta.tex` e `overleaf/resultados.tex` (seções, tabelas,
   contagens, figuras).
7. Branch de trabalho: `v1_proposta_final`.

## 1. O que já sabemos (não repetir — resultados já medidos)

Tudo abaixo é determinístico, do pacote `aprovacao_solida/`, no ANP (182 séries):

* **Busca não-LLM** (12 avaliações, mesmo catálogo): greedy 21,89;
  seeds 21,94; sem_agente 21,97; search_best 22,01; **CREST 22,09**; ucb 22,15;
  random 22,26. Nenhuma comparação pareada vs CREST significativa
  (p > 0,22; greedy 66 melhor / 74 pior / 42 empate). → O agente é
  *indistinguível* de busca simples.
* **Gate de margem** (vencedor vs melhor semente no teste): vencedor 22,09 vs
  semente 21,94; 52 séries a semente ganha, 42 o vencedor, 88 idênticas.
  Margens 0,5–2% dão 22,10–22,12 (piores que os dois); sempre-semente = 21,94.
  Buckets: ganhos de validação >5% do agente pioram +1,14 p.p. no teste
  (winner's curse). → O gate, como testado, **não** é melhoria.
* **Mediana nas pools estáveis** (k=5,7,9): 21,94 → 22,00 (pior); nas 39 séries
  em que o piso muda, 21,91 → 22,21. → Rejeitado por ora.
* **Ablação oficial**: CREST vs sem agente indistinguível (SMAPE p=0,1403;
  POCID p=0,0785; 54,9% das séries com estratégia idêntica).

Conclusão que orienta as etapas: prompt/mais ações não move o ponteiro. Ganho
só pode vir de **(i) escolher melhor o subconjunto** e **(ii) decisão por
horizonte**; e, para o LLM, de **(iii) informação semântica** que a busca não
tem (descrita na Etapa 4).

## 2. Etapa 1 — Seleção gulosa de subconjunto (Caruana, 2004)

**Hipótese**: as pools atuais vêm de regras fixas (`select_stable`, `select_top_k`,
`prune_redundant`); a literatura de ensembles mostra que a **seleção gulosa com
reposição** (forward selection) encontra subconjuntos melhores. Como os pesos
ponderados colapsam para a média (62% das vitórias ponderadas), o ganho está no
subconjunto, não no peso.

**Implementação (offline primeiro)**:
* `melhorias/01_forward_selection/forward_lib.py`
  * `forward_weights(y_true, y_preds, n_steps=20, metric="rmse")`:
    seleção com reposição sobre `(n_windows, n_models, horizon)` achatado;
    a cada passo adiciona o modelo que mais reduz o erro do ensemble corrente;
    retorna `weights = counts / counts.sum()` e o número de passos efetivos.
  * Variantes: métrica `rmse` e `smape`; parada quando nenhum passo melhora;
    `weights_norm` (média simples do subconjunto selecionado) como segundo braço.
* `melhorias/01_forward_selection/run_forward.py`
  * reusa `aprovacao_solida/_common.py` (`v5_config`, `load_context`,
    `detect_drop`) e `pipeline._build_pooled_meta_models` para paridade com a v5;
  * protocolo anti-vazamento: para pontuar a janela `w`, seleciona com as
    janelas `{1,2,3}\{w}` (LOO, igual à seleção de pool); para o teste,
    seleciona com as três;
  * combina com `combiners.combine_weighted`; score agregado com
    `orchestrator_react.metrics.composite_score` usando o âncora
    `state.baseline_aggregate()` e `config.score_weights()`;
  * braços: `forward_wmean` (Caruana), `forward_subset_mean` (média dos
    selecionados), `floor` (melhor das 10 sementes);
  * saída: CSV por série com score de validação e métricas de teste; resumo
    por dataset (igual ao `05_sementes_medianas`).
* `README.md` com comandos por dataset (sem `.sh`);
* `ALTERACOES_NO_TEXTO.md` (preencher só se adotado; rascunho abaixo).

**Critério de adoção**: `forward_*` com score de validação ≤ melhor semente em
>60% das séries E teste confirmando (SMAPE médio menor, contagem de vitórias
favorável). Se confirmar:

**Integração no core (se aprovado)**:
* `orchestrator_react/selection.py`: novo
  `PoolRecipe(method="forward", params={"n_steps": ..., "metric": ...})` e
  branch em `resolve_pool_recipe` (mesma assinatura `(recipe, y_true, y_preds)`
  → índices), para herdar `nested_selection` de graça;
* `orchestrator_react/tools.py`: `select_forward(state, n_steps=20, metric="rmse")`
  (registra a pool e devolve handle, como `select_stable`);
* `orchestrator_react/pool.py`: seed adicional
  `(k, "forward")` ou combinação `weighted(forward)` em `SEED_STABLE_POOLS` —
  avaliar 10→11 sementes;
* flag `ReactConfig.forward_selection: bool = False` para A/B;
* CLI em `run_tsf_orchestrator.py` (`--forward-selection`).

**Alterações no texto se adotado** (detalhar no `ALTERACOES_NO_TEXTO.md`):
* Seção 3.3.3 (Prior de estratégias / conjunto de estratégias fixas):
  "dez estratégias" → "onze", com a nova descrição;
* Tabela de ações de seleção de pool (Seção 3.4): nova linha "Seleção gulosa
  (Caruana)"; se virar ação, "24 ações" → "25" no texto, na figura da
  arquitetura e nas tabelas; **atenção**: a imagem
  `overleaf/crest_arquitetura_2.png` diz "Catálogo 24 ações" → precisa ser
  refeita se a contagem mudar;
* Seção 5.6/5.7 (resultados): nova subseção com o experimento (ver
  `DOCUMENTO_ADICOES.md`).

**Esforço**: ~1 dia. **Risco**: baixo (determinístico).

## 3. Etapa 2 — Combinação/seleção por horizonte

**Hipótese**: o ranking dos preditores muda ao longo do horizonte (há figura de
erro por passo nos resultados). Decisões globais (um pool/peso para todo o
horizonte) perdem isso; escolher por passo pode ganhar.

**Implementação (offline primeiro)**:
* `melhorias/02_por_horizonte/run_per_horizon.py`:
  * braços: `topk_per_horizon` (para cada passo h, média dos k melhores por
    RMSE de validação naquele passo), `best_single_per_horizon` (melhor modelo
    por passo), `forward_per_horizon` (Caruana por passo, se Etapa 1 aprovada);
  * LOO por janela como na Etapa 1; teste com as três;
  * comparar com `floor` e com `forward`/`search_best` da Etapa 1.
* Nota de API: `weights_*` já aceitam `per_horizon=True` e
  `weighting.resolve_recipe` devolve shape `(n_models, horizon)`;
  `combiners.combine_weighted` aceita `(n_models, horizon)`.
  O que **não** existe é pool variável por horizonte — se necessário, o caminho
  mais barato é uma `WeightsRecipe` que zera os modelos fora do top-k de cada
  passo (equivale a um subset por horizonte) em vez de mudar o `ReactState`.
* `README.md` e `ALTERACOES_NO_TEXTO.md`.

**Integração (se aprovado)**:
* `tools.py`: `weights_topk_per_horizon(state, pool, k=3)` e
  `combine_best_per_horizon(state, pool?)` (reintroduz o `best_per_horizon` que
  existia no orquestrador antigo e sumiu do catálogo novo);
* seeds: `best_per_horizon` e `topk_per_horizon(k=3/5)`.

**Texto se adotado**: Tabela de ações de pesos/combinação ganha linhas;
contagem de ações (24→26 se duas); métrica de erro por horizonte já existe e
serve de motivação (citar a figura).

**Esforço**: 1–2 dias. **Risco**: médio (mais superfície para overfit; validar
com o mesmo rigor).

## 4. Etapa 3 — Score e métricas de decisão

**Hipótese**: o score default usa MAPE (peso 0,2) e SMAPE, instáveis com zeros
(ANP) e séries que cruzam zero (ETT). Isso injeta ruído na seleção. Já existe o
preset `scale_free_safe` (a_rmse 0.7, b_smape 0.1, c_mape 0.0, d_pocid 0.2) —
testar é barato.

**Implementação (offline primeiro)**:
* `melhorias/03_score/run_score_sensitivity.py`: reroda o piso determinístico e
  o replay do gate (reusar `aprovacao_solida/02_replay_winner_curse/replay_gate.py`
  com `--score-preset` novo) sob `balanced` vs `scale_free_safe` vs um preset
  novo sem MAPE; comparar **quais estratégias vencem** e o teste.
* Se útil, `ReactConfig.score_per_horizon: bool = False`: agregar o score por
  passo (média dos scores por h) em vez de concatenar tudo — mexer em
  `state._score_metrics`.
* Opção mais forte (se o orientador/usuário quiser): trocar MAPE por **RMSSE**
  ou **MASE** no composite. Envolve `metrics.py` e a Eq. do score no texto.

**Texto se adotado**: Eq. do score (Seção 3.3.1) e seção de métricas
(background) precisam registrar o novo preset/definição; se mudar a métrica, a
comparabilidade com os runs v5 morre e os números precisam ser rerodados.

**Esforço**: 1 dia (preset) a 3 dias (métrica nova + rerun). **Risco**: médio —
muda a definição publicada.

## 5. Etapa 4 — Prompt semântico e batch (o que só o LLM pode usar)

**Hipótese**: o LLM só agrega se receber informação que a busca numérica não
tem. Proibido usar conhecimento externo/calendário (decisão do usuário).
Permitido: **descrever os datasets** e **descrever os modelos** (model cards).

Três sub-ideias independentes (A/B cada uma):

1. **Model cards**: um texto curto por família de preditor (ARIMA/ETS/Theta;
   Random Forest/CatBoost; CWT/DWT/FT; naïve) com vieses e modos de falha,
   injetado no system prompt. Flag `ReactConfig.semantic_cards: bool = False`;
   implementação em `prompts.build_system_prompt`.
2. **Descrição do dataset no turno**: frequência, horizonte, domínio e
   peculiaridades **descritivas** (ex.: "vendas mensais de combustível por
   estado; algumas séries contêm zeros"), fornecidas por um arquivo
   `orchestrator_react/dataset_notes.py` (mapa dataset → 2–3 frases). CLI
   `--dataset-notes/--no-dataset-notes`. Nada de eventos/calendário.
3. **`evaluate_batch`**: nova ação que aceita uma lista de estratégias e
   devolve a observação de cada uma, gastando uma chamada de LLM para várias
   avaliações (reduz latência e aproveita o raciocínio). No
   `registry.TOOLS` + `PERMISSIVE_TOOLS`; contar cada estratégia no orçamento.

**Teste**: subset de 20–30 séries de um dataset, A/B contra a v5 (mesmo
modelo/temperatura); comparar SMAPE pareado, nº de chamadas e tokens (o
`aprovacao_solida/04_latencia_custo` já lê os artefatos novos).
**Texto se adotado**: Seção 3.5/3.6 (prompt do agente) ganha a descrição dos
cards/notas; tabela do catálogo ganha `Avaliação em lote` (24→25) e a figura da
arquitetura precisa ser refeita.

**Esforço**: 2–4 dias (inclui rodadas no servidor). **Risco**: médio-alto —
resultado pode ser nulo de novo; rodar só depois de 1–3.

## 6. Etapa 5 (opcional) — Híbrido LLM-região + busca local

Depois que o LLM escolhe família/pool, o harness roda um greedy local dentro da
região; o LLM revisa. É onde "usar o LLM ao máximo" tem chance real, mas é a
mudança de harness mais invasiva (mexe em `react_loop`/`registry`). Só iniciar
se as etapas anteriores não entregarem ganho.

## 7. Descartado por decisão do usuário (não implementar)

* **Conhecimento externo** (calendário, eventos, notícias): o usuário não
  quer; manter o método autocontido nos dados.
* **Compartilhamento de conhecimento entre datasets** (transfer cross-dataset):
  complica o protocolo experimental e muda a arquitetura; fora do escopo.

## 8. Referências de literatura para justificar as etapas

* Caruana et al. (2004), *Ensemble selection from libraries of models* —
  Etapa 1.
* FFORMA (Montero-Manso et al., 2020) — já é a base do meta-modelo pooled.
* ADE (Sánchez et al.) — já é baseline.
* Hibridização LLM + busca (AutoML/agent surveys) — Etapas 4–5; citar na
  revisão apenas se a Etapa 4 for adotada.
