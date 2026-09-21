# RESUMO — `reasoning=None` vs. `reasoning=low` na série 5 do ETTm2 (N=5 por condição)

Pergunta: a conclusão anterior (que `reasoning=low` causa a regressão de SMAPE da
série 5, trocando `mean pool4` → SMAPE 22,51% por `mean pool3` → 45,31%) sobrevive
a repetição? Protocolo: série 5 do ETTM2, `gpt-oss:20b`, **sem nenhuma das três
flags**, 5 execuções com `reasoning=None` e 5 com `reasoning=low`, mais 2 âncoras
de dataset inteiro.

---

## Aviso de protocolo (importante para ler as tabelas)

Rodar **só** a série 5 (`--indices 5`) remove o **DATASET CARD** do prompt: o card
é o prior leave-one-series-out sobre as *outras* séries do run, e
`_build_strategy_priors` devolve vazio quando o run tem 1 série só. Nos runs
anteriores (publicado, combinado, controles) o card estava presente e continha
recomendações reais para esta série (melhor no dataset: `mean pool1` −0,0137;
piores incluíam `mean`/`dba` e, no combinado, `mean pool3` 0,0132). As 10 execuções
do core rodaram **sem card**; as 2 âncoras rodaram com card (dataset inteiro) para
aferir o efeito do protocolo. Tudo o mais é idêntico (pooled meta-model já era
retido nos dois protocolos — 7 e 1 séries ficam abaixo do mínimo 20; `weights_ols`
idem, 3 janelas < 5). Ordem das execuções: bloco None primeiro, bloco low depois
(**não intercalado** — possível confundimento de ordem/estado do servidor, anotado
na conclusão).

## Tabela das execuções (série 5)

| run | condição | SMAPE teste | estratégia final (origem) | final = `mean pool4`? | avaliou `mean pool4` alguma vez? | parse | handle | respostas vazias | elapsed_s | iters | stop |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `iso_s5_none_01` | None | 45,31% | mean/pool3 (baseline) | não | não | 0 | 0 | 8 | 64,8 | 7 | agent_accepted |
| `iso_s5_none_02` | None | 45,31% | mean/pool3 (baseline) | não | não | 2 | 0 | 16 | 116,4 | 11 | no improvement ×4 |
| `iso_s5_none_03` | None | 45,31% | mean/pool3 (baseline) | não | não | 0 | 0 | 15 | 109,3 | 11 | no improvement ×4 |
| `iso_s5_none_04` | None | 45,31% | mean/pool3 (baseline) | não | não | 2 | 0 | 18 | 119,5 | 11 | no improvement ×4 |
| `iso_s5_none_05` | None | 45,31% | mean/pool3 (baseline) | não | não | 4 | 0 | 26 | 141,6 | 12 | agent_accepted |
| `iso_s5_low_01` | low | 45,31% | mean/pool3 (baseline) | não | não | 0 | 0 | 5 | 31,0 | 8 | no improvement ×4 |
| `iso_s5_low_02` | low | 45,31% | mean/pool3 (baseline) | não | não | 0 | 0 | 2 | 32,1 | 10 | no improvement ×4 |
| `iso_s5_low_03` | low | 22,51% | mean/pool4 (agent) | **sim** | sim | 0 | 0 | 2 | 32,7 | 12 | agent_accepted |
| `iso_s5_low_04` | low | 22,51% | mean/pool4 (agent) | **sim** | sim | 0 | 0 | 2 | 32,4 | 11 | agent_accepted |
| `iso_s5_low_05` | low | 22,51% | mean/pool4 (agent) | **sim** | sim | 0 | 0 | 2 | 29,4 | 10 | no improvement ×4 |
| `iso_s5_anchor_none` (card) | None | 22,51% | mean/pool4 (agent) | sim | sim | 0 | 0 | 10 | 934,0 | 6 | agent_accepted |
| `iso_s5_anchor_low` (card) | low | 45,31% | mean/pool3 (baseline) | não | não | 0 | 2 | 10 | 217,6 | 11 | no improvement ×4 |

"respostas vazias" = erros do tipo `empty response, retrying (N/8)` no
`react.errors` — o modo de falha conhecido do canal de raciocínio do gpt-oss
(documentado em `config.py`; é a razão de existir o `reasoning=low/off`).
"handle" = erros de handle de pesos (`unknown weights handle` / pesos computados
sobre outro pool).

Observação estrutural: o desfecho é praticamente **binário**. O baseline semeado
`mean pool3` é determinístico (mesmos 45,31% em toda execução) e o `mean pool4` do
agente também (22,51%). Não apareceu terceiro desfecho nas 12 execuções.

## Análise

### 1. Taxa de sucesso (final = `mean pool4`, SMAPE 22,51%)

| condição | sucessos | falhas |
|---|---|---|
| None (sem card) | **0/5** | 5/5 |
| low (sem card) | **3/5** | 2/5 |

### 2. Distribuição do SMAPE de teste por condição (core, sem card)

| condição | média | dp | mín | máx | valores |
|---|---|---|---|---|---|
| None | 45,31% | 0,000 | 45,31% | 45,31% | 45,31 ×5 |
| low | 31,63% | 11,171 | 22,51% | 45,31% | 45,31; 45,31; 22,51; 22,51; 22,51 |

### 3. Teste exato de Fisher (sucesso = `mean pool4`)

Tabela `[[None_sucesso, None_falha], [low_sucesso, low_falha]]` = `[[0, 5], [3, 2]]`.

**p = 0,1667** (bicaudal), odds ratio = 0 (nenhum sucesso em `None`; a célula zero
fica no numerador — OR 0, não ∞).

Pelo limiar combinado ("p entre 0,05 e 0,2 → inconclusivo"), este resultado é
**inconclusivo** — e a direção observada favorece o `low`, **o oposto** da
conclusão anterior.

### 4. Tempo (`elapsed_s` do ciclo completo por série, incl. preflight/prepass)

| condição | média | dp | valores |
|---|---|---|---|
| None (série só) | 110,3 s | 25,2 | 64,8 / 116,4 / 109,3 / 119,5 / 141,6 |
| low (série só) | 31,5 s | 1,2 | 31,0 / 32,1 / 32,7 / 32,4 / 29,4 |

Razão None/low ≈ **3,5×** na série isolada. Nas âncoras de dataset inteiro:
934,0 s vs. 217,6 s ≈ **4,3×**. (Consistente com a observação prévia de ~800 s vs
~200 s.) Parte do tempo extra do `None` é o retry das respostas vazias, que é mais
frequente nessa condição.

### 5. Frequência de erros/inconsistências por condição (core)

| tipo de erro | None (total, 5 runs) | low (total, 5 runs) |
|---|---|---|
| parse (ação `unparsed` + `parse_failures`) | 8 (0, 2, 0, 2, 4) | 0 |
| handle de pesos / argumento inválido | 0 | 0 (âncora low: 2) |
| respostas vazias (`empty response, retrying`) | **83** (8, 16, 15, 18, 26) | **13** (5, 2, 2, 2, 2) |

As respostas vazias ocorrem nas duas condições, mas ~6,4× mais no `None`. No
`iso_s5_none_05` uma iteração inteira foi perdida (8/8 retries vazios na iteração
9). Mesmo assim, o agente `None` podou pool3→pool4 em **todas** as 5 execuções e
apenas **nunca avaliou `combine="mean"` sobre pool4** — inclusive em execuções em
que o loop se recuperou dos vazios e sobrou orçamento (ex.: `none_01` computou
pesos no pool podado e em seguida aceitou o baseline `a8` no iter 7, sem tentar a
média). Ou seja: os vazios corroem iterações, mas não explicam sozinhos o 0/5. O
agente `low` avaliou a média do pool podado em 3/5.

## Conclusão (qualificada)

**A conclusão anterior ("`reasoning=low` causa a regressão da série 5") não é
sustentada por este experimento — e os dados apontam na direção oposta no
protocolo sem card.**

- Com N=5 por condição, sem o DATASET CARD: `None` teve **0/5** sucessos e `low`
  **3/5** (Fisher p=0,167 — inconclusivo pelo limiar combinado, mas a direção é
  contrária à do `RESUMO.md` anterior).
- As 2 âncoras com card reproduzem o padrão anterior (None→22,51%, low→45,31%) —
  ou seja, o resultado é **sensível ao protocolo** (presença/ausência do card), e
  o padrão observado antes tem só N=1–2 por célula.
- Há dois mecanismos concorrentes, não um: (a) o movimento estratégico chave
  (avaliar `mean` no pool podado) é encontrado de forma instável nas duas
  condições, em direções que mudam com o prompt; (b) as **respostas vazias** do
  gpt-oss são ~6× mais frequentes com `None` e degradam o loop, o que penaliza o
  `None` independentemente da questão estratégica.
- Confundimentos a anotar: blocos não intercalados (None rodou antes do low;
  efeito de ordem/estado do servidor não pode ser excluído) e ausência do card no
  core.

**Não forçar conclusão.** O que os dados sustentam: a série 5 é instável, o
desfecho é sensível ao card e à taxa de respostas vazias, e a atribuição anterior
a `reasoning=low` tem evidência N=1 que não se confirmou. Para decidir, seria
preciso repetir com mais amostras **intercaladas** (None/low alternados) e,
idealmente, com o protocolo do card (dataset inteiro ou card fixo), antes de
afirmar efeito de qualquer uma das condições.
