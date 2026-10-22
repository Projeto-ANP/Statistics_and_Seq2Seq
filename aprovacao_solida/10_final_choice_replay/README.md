# 10 — Replay offline da seleção final (v11)

Este script testa a decisão final da v11 **sem rerodar o loop ReAct**. Ele lê os
artefatos do v10x, reconstrói a shortlist por família, monta o prompt cego, chama
o seletor e mede o resultado na janela de teste.

O teste real **não entra no prompt**. Ele só é usado depois da escolha, para
avaliar. O prompt vê apenas:

- o perfil da série (passado);
- o último valor observado;
- a mediana e a banda q10–q90 do pool de modelos;
- as previsões futuras dos candidatos (que já existem antes do teste);
- features derivadas dessas previsões.

Não entram no prompt: nome de modelo, tipo de combinação, score de validação,
rank, `attempt_id`, origem, rationale ou valores reais do teste.

## O que o script faz

1. lê `timeseries/mestrado/resultados/<run>/llm_artifacts/<dataset>/dataset_<i>.json`;
2. reconstrói a shortlist: melhor tentativa **plausível** de cada família de
   combinação (`weighted`, `median`, `dba`, `trimmed_mean`, `mean`);
3. remove candidatos com previsão fora de escala (fator configurável);
4. embaralha A/B/C/... de forma determinística;
5. monta o prompt cego e chama o LLM;
6. aplica a política conservadora:
   - `KEEP` → mantém o líder;
   - candidato escolhido com confiança baixa → mantém o líder;
   - previsão degenerada → mantém o líder;
   - parsing inválido → mantém o líder;
7. grava por série: escolha, confiança, motivo, fallback, mapeamento A/B/C,
   teste SMAPE do escolhido, do argmin, do oráculo da shortlist e do oráculo
   global.

## Comandos (servidor)

Rodar da raiz do repositório:

```bash
cd ~/Statistics_and_Seq2Seq
conda activate agno
mkdir -p logs
```

### 0) Testes unitários

```bash
python -m pytest tests/test_final_choice.py -q
```

### 1) Smoke sem LLM (só monta os prompts)

```bash
python3 aprovacao_solida/10_final_choice_replay/replay_final_choice.py \
    --run orchestrator_react_v10x_low \
    --datasets ETTH1 --limit 2 --dry-run \
    --output-dir outputs/resultados/v11_final_choice_replay/dry_run_low
```

Inspecione `prompts.jsonl` para conferir que não há nome de modelo, score, rank,
`attempt_id` ou valor de teste. O mapeamento verdadeiro A/B/C fica em
`mapping.jsonl` (e em `shortlist_specs` no `results.csv`), separado do prompt.
Use um `--output-dir` separado para dry-run/smoke; o run completo usa o default
`outputs/resultados/v11_final_choice_replay/<run>_<model>/`.

### 2) Smoke com LLM (2 séries)

```bash
REACT_OLLAMA_URL=http://127.0.0.1:11435 python3 \
    aprovacao_solida/10_final_choice_replay/replay_final_choice.py \
    --run orchestrator_react_v10x_low \
    --datasets ETTH1 --limit 2 \
    --model gpt-oss:20b --reasoning low --temperature 0.2 --llm-seed 11 \
    --output-dir outputs/resultados/v11_final_choice_replay/smoke_low
```

### 3) Smoke sem servidor (testa a plumbing)

```bash
python3 aprovacao_solida/10_final_choice_replay/replay_final_choice.py \
    --run orchestrator_react_v10x_low \
    --datasets ETTH1 --limit 2 \
    --mock-choice KEEP \
    --output-dir /tmp/v11_final_choice_mock
```

### 4) Replay completo (6 datasets)

```bash
nohup env REACT_OLLAMA_URL=http://127.0.0.1:11435 python3 \
    aprovacao_solida/10_final_choice_replay/replay_final_choice.py \
    --run orchestrator_react_v10x_low \
    --datasets ANP_MONTHLY ETTH1 ETTH2 ETTM1 ETTM2 NN5_WEEKLY_DATASET \
    --model gpt-oss:20b --reasoning low --temperature 0.2 --llm-seed 11 \
    > logs/v11_final_choice_replay_low.log 2>&1 &
```

Para o run medium, troque `--run orchestrator_react_v10x_medium` e o nome do log.

Se o run for interrompido, use `--resume` com o mesmo `--output-dir`:

```bash
REACT_OLLAMA_URL=http://127.0.0.1:11435 python3 \
    aprovacao_solida/10_final_choice_replay/replay_final_choice.py \
    --run orchestrator_react_v10x_low \
    --datasets ANP_MONTHLY ETTH1 ETTH2 ETTM1 ETTM2 NN5_WEEKLY_DATASET \
    --model gpt-oss:20b --reasoning low --temperature 0.2 --llm-seed 11 \
    --resume \
    >> logs/v11_final_choice_replay_low.log 2>&1
```

## Análise dos resultados

Depois do replay, os dois analisadores não chamam LLM:

```bash
# Visão geral: summary, decisões, fallback, qualidade dentro da shortlist
python3 aprovacao_solida/10_final_choice_replay/analyze_replay.py \
    --output-dir outputs/resultados/v11_final_choice_replay/orchestrator_react_v10x_low_gpt-oss-20b \
    --run orchestrator_react_v10x_low

# Regras determinísticas vs LLM (pool median, last, seasonal, median forecast)
python3 aprovacao_solida/10_final_choice_replay/analyze_deep.py \
    --output-dir outputs/resultados/v11_final_choice_replay/orchestrator_react_v10x_low_gpt-oss-20b \
    --run orchestrator_react_v10x_low

# Seasonal guard determinístico, em qualquer run (low e medium)
python3 aprovacao_solida/10_final_choice_replay/analyze_seasonal_rule.py \
    --run orchestrator_react_v10x_low
python3 aprovacao_solida/10_final_choice_replay/analyze_seasonal_rule.py \
    --run orchestrator_react_v10x_medium
```

## Saídas

No diretório do run:

- `results.csv` — uma linha por série, com as métricas pós-escolha;
- `summary.csv` — média por dataset e média dos datasets;
- `prompts.jsonl` — **somente** o system/user enviados ao modelo, a resposta
  bruta e a resposta parseada; nenhum nome/score/rank;
- `mapping.jsonl` — auditoria completa por série: mapeamento A/B/C, specs,
  scores de validação e features (não é enviado ao modelo);
- `run_command.txt` — comando exato, `cwd` e `base_url`.

Colunas principais do `results.csv`:

- `argmin_smape` — o v10 (referência);
- `chosen_smape` — o que a v11 aplicou;
- `shortlist_oracle_smape` — melhor possível dentro da shortlist (teto local);
- `top3_oracle_smape` — melhor possível no top3 de validação;
- `global_oracle_smape` — melhor possível em todas as tentativas (teto global);
- `override_rate`, `keep_rate`, `fallback_rate`;
- `win_rate`, `loss_rate`, `tie_rate` contra o argmin;
- `mean_gain_vs_argmin`.

## Comparação três-a-três (v0.1)

Compara, nas mesmas trajetórias do v10x:

1. `argmin` (v10);
2. `seasonal_guard` determinístico;
3. `seasonal_guard` + veto binário do LLM.

O LLM só é chamado quando o guard quer trocar (≈15% das séries). Primeiro sem
LLM, para conferir o guard e os prompts:

```bash
python3 aprovacao_solida/10_final_choice_replay/compare_three.py \
    --run orchestrator_react_v10x_low --dry-run
python3 aprovacao_solida/10_final_choice_replay/compare_three.py \
    --run orchestrator_react_v10x_medium --dry-run
```

Depois com o veto real:

```bash
nohup env REACT_OLLAMA_URL=http://127.0.0.1:11435 python3 \
    aprovacao_solida/10_final_choice_replay/compare_three.py \
    --run orchestrator_react_v10x_low \
    --model gpt-oss:20b --reasoning low --temperature 0.2 --llm-seed 11 \
    > logs/v11_three_way_low.log 2>&1 &
```

Repita para `orchestrator_react_v10x_medium`.

Saídas em `outputs/resultados/v11_three_way/<run>_<model>/` (o dry-run usa
`<run>_<model>_dry_run/`, para não misturar com a run real):

- `summary.csv` — argmin × guard × veto por dataset;
- `results.csv` — por série, com `guard_gain_vs_argmin`,
  `veto_gain_vs_guard`, `veto_correct`, `veto_reason`, etc.;
- `prompts.jsonl` — só o prompt/resposta do veto;
- `mapping.jsonl` — mapeamento A/B, leader/seasonal e métricas.

Métricas que importam:

- `veto_gain_vs_guard`: positivo = o LLM melhorou sobre o guard;
- `veto_switch_rate`: em quantas chamadas o LLM aceitou a troca sazonal;
- `veto_accuracy`: em quantas chamadas o LLM escolheu o melhor dos dois
  candidatos (líder vs seasonal) no teste;
- `veto_correct` é só para análise; não entra na decisão.

Depois disso, o pipeline completo pode rodar com:

```bash
--final-strategy seasonal_guard
# ou
--final-strategy seasonal_veto --final-selector-temperature 0.2
```

## Comando do pipeline real (smoke)

Depois de validar o replay, rode o pipeline de ponta a ponta em duas séries:

```bash
python3 run_tsf_orchestrator.py \
    --dataset ETTH1 --source ETTH1.tsf --source-dir ../forecasting_datasets \
    --combinator gpt-oss:20b --version v11_smoke --gpu 1 \
    --reasoning low --llm-seed 11 --prompt-v2 --catalog-v2 \
    --no-early-stop --scoring-budget --max-iterations 12 --max-total-turns 24 \
    --temperature 0.7 \
    --score-preset smape_only --score-window-mode weighted \
    --no-agent-score-preset --no-seed-attempts --no-best-single \
    --final-strategy agent_family \
    --final-selector-model gpt-oss:20b \
    --final-selector-temperature 0.2 \
    --final-selector-reasoning low \
    --final-selector-seed 11 \
    --indices 0 1
```

O artefato de cada série terá o bloco `final_choice` com:

- `prompt.system` e `prompt.user`;
- `response`;
- `parsed` (`choice`, `confidence`, `reason`);
- `shortlist` com o mapeamento A/B/C → `attempt_id`, spec e score real;
- `decision` (`keep`, `override`, `fallback`);
- `fallback` e `chosen_id`.

## Notas

- O `--limit` é por dataset.
- O `--indices` seleciona índices específicos; combinado com `--limit`, o limite
  é aplicado depois do filtro.
- `--no-keep` remove a opção `KEEP` (ablação).
- `--no-low-confidence-keep` permite override com confiança baixa (ablação).
- `--degenerate-factor` controla o corte de previsões fora de escala (default
  100× a escala histórica).
- O script usa `REACT_OLLAMA_URL` se `--base-url` não for passado.
- O prompt é em inglês, como o restante do harness; o documento e os logs são
  em português.
- O replay mede no teste. Para ajustar o prompt, use um subconjunto de séries
  para desenvolvimento e avalie no restante (por exemplo, `--indices` em um
  split fixo); escolher o prompt olhando o resultado de todas as séries é
  overfitting no teste.
