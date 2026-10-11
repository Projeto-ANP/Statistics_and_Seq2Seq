# 01 — Busca não-LLM sobre o mesmo catálogo (isola o efeito do LLM)

**Pergunta**: com o MESMO catálogo de ações e o MESMO orçamento (12 avaliações),
uma busca simples — aleatória, gulosa ou bandit (UCB) — empata ou vence o agente
LLM? Se vencer, o que não agrega é a busca em geral, e o LLM é só um search caro.
Se o agente vencer, a decisão do LLM tem mérito próprio.

Tudo determinístico, sem Ollama. Não altera nenhum resultado existente: grava em
`timeseries/mestrado/resultados/orchestrator_search_<version>/`.

## Comandos (copiar e colar no servidor)

```bash
conda activate agno
cd ~/Statistics_and_Seq2Seq
mkdir -p logs

# 1) SMOKE: 3 series de um dataset, os 4 braços (~1 min)
python3 aprovacao_solida/01_busca_nao_llm/run_search.py \
    --dataset ANP_MONTHLY --indices 0 1 2 --version v1 \
    --methods seeds,random,greedy,ucb

# 2) UM dataset completo
nohup python3 aprovacao_solida/01_busca_nao_llm/run_search.py \
    --dataset ANP_MONTHLY --version v1 \
    > logs/search_v1_anp.log 2>&1 &

# 3) TODOS os datasets, em sequência (cole o bloco inteiro; não precisa de .sh)
nohup bash -c 'for ds in ANP_MONTHLY NN5_WEEKLY_DATASET M4_WEEKLY_DATASET ETTH1 ETTH2 ETTM1 ETTM2; do
    python3 aprovacao_solida/01_busca_nao_llm/run_search.py --dataset "$ds" \
        --version v1 --methods seeds,random,greedy,ucb
done' > logs/search_v1_all.log 2>&1 &

# 4) Análise (roda depois de terminar; não precisa de LLM)
python3 aprovacao_solida/01_busca_nao_llm/analyze_search.py --version v1
```

## O que sai

* `timeseries/mestrado/resultados/orchestrator_search_v1/<DATASET>.csv`
  — uma linha por (série, braço), `regressor = search_<seeds|random|greedy|ucb>`.
* `.../orchestrator_search_v1/search_artifacts/<DATASET>/dataset_<i>_<braço>.json`
  — todas as tentativas de cada braço (spec + score), para inspeção.
* `outputs/resultados/aprovacao_solida/analise_busca.csv` (análise pareada).

## Como ler o resultado

No `analyze_search.py`, compare, por dataset, `search_best` (o melhor braço por
*série*, escolhido pelo score de validação — um teto para busca simples) com
`CREST` e `sem_agente`:

* `search_best` vence/empata `CREST` → **o LLM não agrega**; a contribuição da
  dissertação é o framework + o meta-learning entre séries + a abstenção de
  busca. Frase defensável: "nem o melhor de três buscas simples supera o piso
  determinístico".
* `CREST` vence `search_best` de forma consistente → o LLM agrega valor de
  decisão; aí sim vale investigar onde (poucas séries? evidência escassa?).

Notas de honestidade metodológica (coloque no texto):

* a busca ganha 12 *avaliações*; o agente ganha 12 *iterações* (algumas gastas
  em ferramentas que não pontuam). Isso é levemente generoso com a busca — o que
  só fortalece a conclusão se a busca perder.
* `greedy` pode parar antes do orçamento quando nenhum vizinho melhora; o número
  de avaliações gastas fica registrado em `n_evals`.
* o braço `seeds` é o piso (sem busca nenhuma) e deve reproduzir, série a série,
  o `orchestrator_baseline_v1` (mesmo código e sementes).
