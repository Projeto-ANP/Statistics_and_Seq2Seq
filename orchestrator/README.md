# Orchestrator (Agno) – multi-agent research loop + deterministic evaluator

This folder implements the multi-agent architecture you described:

- **Proposer**: proposes candidate combination strategies (strict JSON).
- **Skeptic/Auditor**: removes leakage / invalid validation schemes.
- **Statistician**: adds robustness (top-k, shrinkage, ridge).
- **Evaluator (deterministic, code)**: runs rolling/expanding evaluation and computes **MAPE, sMAPE, RMSE, POCID** per-horizon and aggregate.

## Data contract (already in your context)

The evaluator reads from `agent.context.CONTEXT_MEMORY["all_validations"]`:

```python
{
  "predictions": [
    {"ARIMA": [h1,h2,...], "ETS": [...], ...},  # window 0
    {"ARIMA": [h1,h2,...], "ETS": [...], ...},  # window 1
    ...
  ],
  "test": [
    [y1,y2,...],  # window 0
    [y1,y2,...],  # window 1
    ...
  ]
}
```

## Supported strategy toolbox

Set `params.method` to one of:

- `mean`
- `median`
- `trimmed_mean` (needs `params.trim_ratio`)
- `best_single` (rolling selection)
- `best_per_horizon` (rolling selection)
- `topk_mean_per_horizon` (needs `params.top_k`)
- `inverse_rmse_weights_per_horizon` (needs `params.top_k`, optional `shrinkage`)
- `ridge_stacking_per_horizon` (needs `params.l2`, optional `top_k`)

All selection/weighting strategies are evaluated in an **anti-leakage** way: at window `i`, they fit/select using only windows `< i`.

## Run

From `Statistics_and_Seq2Seq/`:

```bash
python -m orchestrator.run_research_loop \
  --dataset-index 146 \
  --models ARIMA,ETS,THETA \
  --ollama-model qwen3:14b \
  --rolling expanding \
  --train-window 5
```

- Add `--use-llm` to run the proposer/skeptic/statistician debate. Without it, it evaluates a solid default candidate set.

## Run (same style as `run_tsf_agents.py`)

If you want the same "loop over dataset_index and save a CSV" workflow, use:

```bash
conda activate agno
cd /home/anp/Documents/lucas_mestrado/Statistics_and_Seq2Seq
python run_tsf_orchestrator.py
```

This script mirrors the structure of `run_tsf_agents.py`: it builds the context per dataset, runs the orchestrator deterministically, produces final predictions for the chosen strategy, computes MAPE/sMAPE/RMSE/POCID, and appends rows to a results CSV.

## Outputs

- Tool output is stored in context key: `orchestrator_last_eval`
- Ranking includes aggregate + stability + a composite score normalized by the baseline mean.
