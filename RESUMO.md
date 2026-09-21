# RESUMO — Decomposição do efeito reordenação/catálogo/semeadura no ETTm2 e regressão de SMAPE

Comparação central: run publicado `orchestrator_react_v5` (SMAPE 16,10% / POCID 45,34%) vs.
run combinado `orchestrator_react_v2_gpt_low_nivel` (SMAPE 19,34% / POCID 45,34%) — ETTM2, 7 séries,
combinador `gpt-oss:20b`.

---

## Passo 0 — Configuração do run original publicado (`orchestrator_react_v5`, ETTM2)

**O run publicado NÃO usou `--reasoning low`.** O config serializado nos artefatos
(`llm_artifacts/ETTM2/dataset_*.json`, campo `config`) registra, nas 7 séries:

- `combinator.reasoning: None` → o default do servidor Ollama para o gpt-oss
  (que para esse modelo é reasoning **ligado, budget default**). `--reasoning low`
  *capa* o budget — é outra coisa.
- Nenhuma das três flags presente no config (`reorder_weight_tools`,
  `drop_redundant_combine_actions`, `reduced_seeding` ausentes; `asdict` inclui
  todo campo do schema — a ausência no JSON salvo significa que o código que
  gerou o v5 era anterior a esses campos, que são recentes).
- Demais campos idênticos aos defaults atuais (`seed_stable_pools=True`,
  `dataset_card=True`, `pool_mode=full`, `final_strategy=argmin`,
  `backtest_mode=expanding`, `n_validation_windows=3`, `score_preset=balanced`).
  Observação: o campo `prompt_format` não existia no schema do v5; o default
  atual é `xml`, que é o formato que o v5 já usava (o toggle foi introduzido
  depois com esse default).

Verifiquei também os demais datasets do mesmo run publicado (ANP 182 séries, ETTH1/2,
ETTM1/2, M4 359, NN5 111): **todos** têm `reasoning: None` e nenhuma flag — a premissa
"v5 usou `--reasoning low`" é falsa em todo o run publicado.

**Consequência:** o run combinado `v2_gpt_low_nivel` mudou **quatro** variáveis em
relação ao publicado (não três): as três flags **e** `reasoning: None → low`
(config salvo confirma `'reasoning': 'low'`). O Passo 1 foi então desenhado com
um braço a mais, `iso_reasoning_low`, para isolar essa quarta variável — sem ele
a decomposição da regressão ficaria incompleta.

(Provável origem da premissa errada: os `RESUMO_*.md` em `outputs/resultados/ablacao/`
sugerem comandos de validação com `--reasoning low`; os artefatos do run publicado é
que são a fonte definitiva.)

---

## Passo 1 — Isolamento das variáveis no ETTM2 (7 séries, `gpt-oss:20b`)

Desenho final (5 runs, todos só no ETTM2):

- `iso_control_repeat` — **sem** `--reasoning` (None, igual ao publicado), nenhuma
  flag: reexecução da configuração original, para medir a variação entre execuções.
- `iso_reasoning_low` — só `--reasoning low`: isola a 4ª variável que o combinado
  também mudou (None→low) e serve de **controle** para os braços de flag abaixo.
- Os 3 braços de flag rodam **com** `--reasoning low` (decisão do autor, para
  custar ~200s cada em vez de ~800s): cada flag é medida no mesmo contexto do run
  combinado, variando uma coisa só em relação ao `iso_reasoning_low`.

| Run | Config vs. controle | SMAPE | POCID |
|---|---|---|---|
| `orchestrator_react_v5` (publicado) | — (referência) | **16,10%** | **45,34%** |
| `orchestrator_react_v2_gpt_low_nivel` (combinado) | 3 flags + `--reasoning low` | **19,34%** | **45,34%** |
| `iso_control_repeat` | nenhuma (reexecuta o publicado) | **16,08%** | **45,34%** |
| `iso_reasoning_low` | só `--reasoning low` | **19,33%** | **45,34%** |
| `iso_reorder_low` | `low` + só `--reorder-weight-tools` | **16,08%** | **45,34%** |
| `iso_dropcomb_low` | `low` + só `--drop-redundant-combine` | **20,30%** | **45,34%** |
| `iso_reducedseed_low` | `low` + só `--reduced-seeding` | **19,39%** | **45,34%** |

Logs: `logs/iso_*_ettm2.log` (e `logs/isolation*_batch.log`).

### Leitura dos números (com o cuidado que o desenho exige)

- **Controle reproduz o publicado** (16,08% ≈ 16,10%; POCID idêntico): a variação
  entre execuções é desprezível, então as comparações contra 16,10% são válidas.
- **`--reasoning low` sozinho reproduz a regressão inteira**: 19,33% ≈ 19,34% do
  combinado. A série 5 caiu no baseline `mean pool3` (45,31%) neste braço, no
  `iso_dropcomb_low`, no `iso_reducedseed_low` e no combinado — e **não** caiu no
  controle nem no publicado (que acham `mean pool4`, 22,51%). A única exceção é o
  `iso_reorder_low`, onde o agente com `low` **achou** o `mean pool4` (iter 5,
  score 0,0434) — ou seja, o efeito do `low` é estocástico, não determinístico:
  em 3 de 4 runs com `low` o agente não avalia a média do pool podado e cai no
  piso; em 1 de 4 avalia e vence (como no publicado). `low` desloca a
  probabilidade de o agente fazer esse movimento, não a zera.
- **`reorder_weight_tools`: nenhum efeito.** 16,08%, idêntico ao controle série a
  série (e foi justamente o run em que o `low` não derrubou a série 5 — coerente
  com a leitura estocástica acima, não com efeito da reordenação).
- **`drop_redundant_combine_actions` (em `low`): 20,30%.** A série 5 caiu no piso
  como em todos os runs `low` (45,31%); além disso a série 0 caiu para o baseline
  `trimmed_mean pool3` (19,83%) após um erro de parse no iter 5. Essa queda da
  série 0 **não** ocorreu no combinado (que tinha a mesma flag + `low` e ficou em
  13,10% com origem `agent`) — é ruído de execução, não efeito sistemático da
  flag. Sem evidência de efeito adicional da flag além do comportamento
  compartilhado do `low`.
- **`reduced_seeding` (em `low`): 19,39%.** Mesma queda da série 5 (45,31%);
  série 0 em 13,43% (mesma estratégia, outro handle de pesos — dentro do ruído).
  Sem evidência de efeito adicional.

**Conclusão do Passo 1:** a regressão de SMAPE do combinado é atribuível ao
`--reasoning low` (que o combinado introduziu junto com as flags; o publicado não
o usava). Nenhuma das três flags, sozinha, reproduz a regressão além do que o
`low` já produz; entre as três, `reorder_weight_tools` mostrou efeito nulo e
`drop_redundant_combine_actions`/`reduced_seeding` não mostraram efeito
sistemático adicional (os desvios observados nelas são estocásticos, com N=1 por
braço). Ressalva do desenho, dita explicitamente: os braços de flag foram medidos
**com** `low` (decisão do autor, ~200s/run); o efeito de cada flag *sem* `low`
não foi medido, mas o controle (sem flags, sem `low`) e o braço `low` (sem flags)
delimitam que a diferença publicado→combinado passa pelo reasoning.

---

## Passo 2 — Séries que pioraram (v5 vs. combinado), série a série

| série | SMAPE v5 | SMAPE comb. | Δ | estratégia v5 (origem) | estratégia combinado (origem) |
|---|---|---|---|---|---|
| 0 | 13,06% | 13,10% | +0,04 p.p. | weighted pool4, w=inverse_error (agent) | weighted pool4, w=softmax_neg_error (agent) |
| 1 | 26,53% | 26,53% | 0 (idêntico) | mean pool1 (baseline) | mean pool1 (baseline) |
| 2 | 15,69% | 15,69% | 0 (idêntico) | mean pool2 (baseline) | mean pool2 (baseline) |
| 3 | 25,51% | 25,51% | 0 (idêntico) | trimmed_mean pool1 (baseline) | trimmed_mean pool1 (baseline) |
| 4 | 7,41% | 7,41% | 0 (idêntico) | trimmed_mean pool3 (baseline) | trimmed_mean pool3 (baseline) |
| **5** | **22,51%** | **45,31%** | **+22,80 p.p.** | **mean pool4 (agent)** | **mean pool3 (baseline)** |
| 6 | 1,97% | 1,83% | −0,14 p.p. | best_single CWT_rf (agent) | mean pool1 (baseline) |

Médias: 16,0964% → 19,3395% (+3,24 p.p.). A regressão total (3,24 p.p.) é
**inteiramente** da série 5 (22,80/7 ≈ 3,26, compensada em −0,02 pelas séries 0 e 6).
As séries 1–4 têm SMAPE **idêntico dígito a dígito** (mesma estratégia, mesmos
números). POCID não muda em nenhuma série (45,34% nos dois runs).

### Série 5 — o mecanismo da piora

- Nos **dois** runs o agente podou `pool3` → `pool4` com `prune_redundant`, obtendo
  exatamente os mesmos 4 modelos (`ARIMA, DWT_rf, FT_rf, ONLY_CWT_rf`).
- **v5 (publicado):** na iteração 4 o agente avaliou `evaluate_strategy(combine="mean",
  pool="pool4")` → score 0,0434, o melhor do run (o piso semeado era mean pool3,
  score 0,0447). Venceu por margem mínima → SMAPE teste 22,51%.
- **combinado:** o agente podou o mesmo pool, mas **nunca avaliou a média simples sobre
  pool4** — só combinações *ponderadas* sobre pool4 (scores 0,0521–0,0535, todas piores
  que o piso 0,0447). Nada venceu o piso → *early stop* por "no improvement in 4
  consecutive proposals" → caiu no **baseline semeado** `mean pool3` → SMAPE teste
  45,31%. Origem mudou de `agent` para `baseline`.
- O piso semeado era o mesmo nos dois runs (mean pool3, score 0,0447): `reduced_seeding`
  remove mean/median do pool **completo** (que pontuavam 0,7246/0,0566 nesta série e
  não eram o piso), não os pools estáveis. A piora não veio de um piso rebaixado — veio
  de o agente não ter achado a melhoria de margem mínima que achou no v5.
- O que mudou entre as duas trajetórias (o agente não ter tentado `combine="mean"` em
  pool4) foi resolvido pelo Passo 1: é o **`--reasoning low`** — nenhuma das três
  flags reproduz a queda sozinha. No `iso_reasoning_low` o agente cometeu até erros
  de contabilidade de pesos que desperdiçam iterações (handle `w2` inexistente;
  pesos computados sobre pool3 usados com pool4), sintoma do orçamento de raciocínio
  capado. O efeito é estocástico: em 1 de 4 runs com `low` (`iso_reorder_low`) o
  agente avaliou a média do pool podado e achou o 22,51%.
- Séries 0 e 6: diferenças pequenas e em direções opostas (0 piora 0,04 p.p.
  trocando inverse_error por softmax_neg_error — o Passo 1 mostrou que a
  reordenação não muda o SMAPE, então isso é ruído; 6 melhora 0,14 p.p. porque o
  combinado caiu num baseline que acertou o teste).

---

## Passo 3 — A nota "this strategy is numerically the MEAN of its own pool"

A nota aparece no log do combinado para **4 séries: 1, 2, 5 e 6** (linhas após cada
`TEST`; coincide exatamente com as 4 séries cujo `decision.reducibility` tem
`equivalent_to_pool_mean=true`, `pool_mean_relative_diff=0.0`). As estratégias dessas
4 séries no combinado são todas `combine="mean"` sobre um pool estável — ou seja, a
nota é literalmente verdadeira (a estratégia **é** a média do próprio pool, não apenas
"numericamente equivalente").

**A série 5 é uma das séries com a nota — e é exatamente a única série que piorou.**
A média em questão é `mean pool3` (o pool estável de k=9), que é uma **semente de
estabilidade**, não a semente de mean sobre o pool completo que `reduced_seeding`
removeu. Confirma-se a hipótese levantada: a flag remove a semente de mean do pool
completo, mas o piso do run **continua convergindo para uma média por outro caminho**
(os pools estáveis semeiam `mean pool=k` para k∈{5,7,9}), e é essa média que caiu no
teste com SMAPE 45,31%. No v5, com a mesma semente disponível, o agente escapou dela
por 0,0013 de score (0,0434 vs 0,0447) avaliando a média do pool podado — e a margem
teste era enorme (22,51% vs 45,31%).
