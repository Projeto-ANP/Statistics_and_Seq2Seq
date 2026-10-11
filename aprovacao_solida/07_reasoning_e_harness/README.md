# 07 — Rerodar com reasoning controlado (só depois dos itens 01–06)

**Decisão que este item documenta**: vale rerodar tudo com `--reasoning low`?
A resposta curta é **não às cegas** — faça um A/B de subconjunto primeiro; se o
agente empatar (esperado) e a latência cair, rode a versão final com
`--reasoning low` e, de preferência, `--final-strategy seed_gate`.

## O que já está pronto na branch `v1_proposta_final`

1. **Reasoning de verdade**: `--reasoning off|low|medium|high` passa o nível
   direto ao Ollama (`think="low"` etc.). Antes, `bool("low")` virava `True` e o
   nível era ignorado (só `off` funcionava).
2. **Fallback**: no erro 500 "error parsing tool call" (canal harmony do
   gpt-oss), repete o turno uma vez com `think=False` em vez de derrubar a série.
3. **Retry útil**: erros de LLM agora variam o seed (antes repetiam a mesma
   falha determinística).
4. **Reasoning salvo nos artefatos**: cada turno grava `ollama_thinking_text`
   (até 12k chars) em `react.step_details`, além de `thinking_chars`. Dá para
   analisar *por que* o agente propôs cada estratégia.
5. **`seed_gate`**: `--final-strategy seed_gate --seed-gate-margin 0.01` mantém a
   melhor semente a menos que o agente melhore o score de validação em 1% ou
   mais. É a proteção contra winner's curse; use o item 02 para escolher o
   margem com dados antes de gastar GPU.

## Passo 1 — A/B de reasoning num subconjunto (obrigatório antes de rerodar tudo)

```bash
conda activate agno
cd ~/Statistics_and_Seq2Seq
mkdir -p logs

# A) subconjunto, reasoning como sempre foi (default do modelo = ligado)
nohup python3 run_tsf_orchestrator.py \
    --dataset ANP_MONTHLY --source mes_11_venda_mensal.tsf \
    --combinator gpt-oss:20b --version v6_reason_default --limit 20 \
    > logs/v6_reason_default.log 2>&1 &

# B) MESMO subconjunto, reasoning low
nohup python3 run_tsf_orchestrator.py \
    --dataset ANP_MONTHLY --source mes_11_venda_mensal.tsf \
    --combinator gpt-oss:20b --version v6_reason_low --limit 20 --reasoning low \
    > logs/v6_reason_low.log 2>&1 &
```

Compare latência/custo (esquema novo grava tudo):

```bash
python3 aprovacao_solida/04_latencia_custo/analyze_latency.py \
    --versions orchestrator_react_v6_reason_default orchestrator_react_v6_reason_low \
    --datasets ANP_MONTHLY
```

Compare acurácia no mesmo subconjunto (SMAPE pareado):

```bash
python3 aprovacao_solida/03_estatistica_por_dataset/stats_por_dataset.py \
    --datasets ANP_MONTHLY --reference CREST \
    --extra CREST_low=orchestrator_react_v6_reason_low \
            CREST_default=orchestrator_react_v6_reason_default
```

Regra de decisão:

* SMAPE empatado (dentro do ruído) **e** latência menor → rerode a versão final
  com `--reasoning low`.
* SMAPE pior de forma clara com `low` → mantenha os resultados atuais e reporte
  o A/B como análise de custo; não rerode.

## Passo 2 — Rerodar a versão final (somente se o passo 1 autorizar)

Uma linha por todos os datasets, com `seed_gate`, mais iterações e reasoning low:

```bash
nohup python3 run_tsf_batch.py \
    --datasets ANP_MONTHLY NN5_WEEKLY_DATASET M4_WEEKLY_DATASET ETTH1 ETTH2 ETTM1 ETTM2 \
    --combinators gpt-oss:20b --version v6_final \
    -- --reasoning low --final-strategy seed_gate --seed-gate-margin 0.01 \
       --max-iterations 20 \
    > logs/v6_final_batch.log 2>&1 &
```

Depois compare com a v5 usando os itens 03 e 04:

```bash
python3 aprovacao_solida/03_estatistica_por_dataset/stats_por_dataset.py \
    --search-version v1 \
    --extra CREST_v6=orchestrator_react_v6_final \
    > outputs/resultados/aprovacao_solida/v6_vs_v5.txt

python3 aprovacao_solida/04_latencia_custo/analyze_latency.py \
    --versions orchestrator_react_v5 orchestrator_react_v6_final
```

## Regras de higiene experimental

* **Nunca misture configurações** entre braços comparados (reasoning, gate,
  iterações). Se a v6 é o novo resultado principal, ela substitui a v5 por
  inteiro, e o texto registra a diferença.
* O `seed_gate` só troca a decisão quando o agente **não** melhora a semente por
  ≥ margem; ele não pode piorar o resultado em teste de forma sistemática além
  do que o item 02 já mede.
* Se escolher rodar com `seed_gate`, rode também uma cópia sem gate no mesmo
  subconjunto (`--final-strategy argmin`) para reportar o A/B do gate — 20
  séries bastam.
* Guarde o log: `--reasoning low` é a primeira coisa a citar no capítulo de
  custo; sem o A/B, "reasoning low" é só afirmação.
