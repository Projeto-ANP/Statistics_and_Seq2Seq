# 04 — Latência e custo do agente

**Pergunta**: quanto custa (tempo, tokens, reasoning) rodar o agente, e quanto
disso é reasoning? Serve para a seção de custo-benefício ("o agente custa X e
não melhora SMAPE") e para decidir o rerodada com `--reasoning low`.

Lê os artefatos (`llm_artifacts/...`), sem rodar nada.

## Comandos (copiar e colar no servidor)

```bash
conda activate agno
cd ~/Statistics_and_Seq2Seq

# Runs que já existem
python3 aprovacao_solida/04_latencia_custo/analyze_latency.py \
    --versions orchestrator_react_v5 orchestrator_react_v5_qwen

# Depois do A/B de reasoning (item 06), compare as duas versões:
python3 aprovacao_solida/04_latencia_custo/analyze_latency.py \
    --versions orchestrator_react_v5 orchestrator_react_v6_reasoning_low
```

## O que sai

* Tabela no terminal: por dataset, `iter`, `calls`, `s/call`, `llm_s`, `tot_h`,
  `think_k` (milhares de chars de reasoning), tokens de prompt e de avaliação,
  vazias, retries de erro, fallbacks de `think=False`.
* `outputs/resultados/aprovacao_solida/latencia_custo.csv`.

## Atenção aos esquemas de artefato

* Runs antigos (v5) **não** gravaram `step_details`: a coluna `artifacts_schema`
  sai `legacy` e tokens/reasoning aparecem como 0. Tempo total e por chamada
  continuam corretos (vêm do `trajectory`).
* Runs novos (`v1_proposta_final`, que salva o texto de reasoning) saem `full`:
  tokens, `thinking_kchars`, `done_reason` e fallsbacks são reais. **É só com
  esses que o A/B de reasoning low tem sentido quantitativo.**
