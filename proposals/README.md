# Propostas D1–D3 — LAYA dentro do ReAct

Três desenhos, três arquivos independentes, mesma seleção final (argmin —
melhor score de validação do histórico, padrão v5) para comparar SÓ a
estrutura do loop. O gate LAYA zero-shot por janela roda como DADO de
análise (gravado nos CSVs/artifacts), não como juiz.
Nenhum prompt/gate menciona piso/baseline — as sementes são apenas candidatos
no histórico.

| arquivo | desenho | quem decide a ação | custo por turno |
|---|---|---|---|
| `d1_plan_verify.py` | **Plano + verificação** (ReWOO × VEGAS) | LAYA: shortlist determinística + 3 perguntas (move/call/gain) em 1 passada | 1 passada LAYA |
| `d2_verifier.py` | **Verificador por etapa** (VEGAS) | gpt-oss propõe (ReAct padrão); LAYA verifica a ação ANTES de executar | 1 chamada gpt-oss + 1 LAYA |
| `d3_router.py` | **Roteamento por turno** (Switchcraft) | LAYA roteia: rotina → LAYA escolhe da shortlist; exploração → gpt-oss propõe | rotina: 1 LAYA; exploração: 1 gpt-oss |

## Como rodar (no servidor, conda env `agno`)

```bash
# D1 — puro LAYA (não usa o gpt-oss)
nohup python3 proposals/d1_plan_verify.py --datasets ETTH1 --version v1 \
  > logs/d1_v1_etth1.log 2>&1 &

# D2 — gpt-oss + verificador LAYA
nohup python3 proposals/d2_verifier.py --datasets ETTH1 --version v1 --reasoning low \
  > logs/d2_v1_etth1.log 2>&1 &

# D3 — roteador LAYA + gpt-oss
nohup python3 proposals/d3_router.py --datasets ETTH1 --version v1 --reasoning low \
  > logs/d3_v1_etth1.log 2>&1 &

tail -n 30 -f logs/d1_v1_etth1.log
```

Flags comuns: `--indices 0 1 2` (subconjunto de séries), `--no-dataset-card`,
`--max-iterations N`, `--datasets ETTH1 NN5_WEEKLY_DATASET` (vários datasets,
um comando por braço).

Monitorar/kill (sempre por versão):

```bash
ps aux | grep -E "d1_plan_verify|d2_verifier|d3_router" | grep -v grep
pkill -f "d1_plan_verify.py --version v1"
```

## Saídas

- CSV: `timeseries/mestrado/resultados/orchestrator_d{1,2,3}_<version>/<dataset>.csv`
  (mesmas colunas do A2; coluna `seed_floor_smape` = piso das sementes; o bloco
  final do log imprime FFORMA/ADE/median/baseline).
- Artifacts por série: `.../llm_artifacts/<dataset>/dataset_<idx>.json`
  (turnos, respostas do LAYA, veredito, rejeições, gate completo).
- Logs: `logs/d{1,2,3}_*.log` (turno a turno).

## Comparação local (sem LLM, determinística)

```bash
python3 - <<'EOF'
import pandas as pd, glob, json
import numpy as np
BASE='timeseries/mestrado/resultados'
refs = ['median', 'FFORMA', 'ADE']
for folder in sorted(glob.glob(f'{BASE}/orchestrator_d[123]_*')):
    for csv in sorted(glob.glob(f'{folder}/*.csv')):
        df=pd.read_csv(csv, sep=';')
        print(f"{folder.split('_')[-2:]}: {df.smape.mean():.4f} (n={len(df)})")
EOF
```

## Referências

- ReWOO — Xu et al. 2023 (arXiv:2305.18323): plano antes de observar.
- VEGAS — "Think Twice, Act Once" (arXiv:2605.12620): verificador pré-execução.
- ToolVerifier — Meta, EMNLP 2024: verificação da escolha de ferramenta.
- Tool retrieval — "How Many Tools Should an LLM Agent See?" (arXiv:2605.24660):
  shortlist de ferramentas em vez do catálogo completo.
- Switchcraft — Microsoft 2026: roteamento por turno para agentic tool calling.
- Agent-as-a-Router (arXiv:2606.22902): loop Contexto-Ação-Feedback verificado.
- RLVR para tool-use (arXiv:2607.01465): recompensa verificável por ação
  (o score aninhado do backtest é exatamente isso).
- MCS — Hansen, Lunde & James 2011 (Econometrica): seleção entre empatados.
