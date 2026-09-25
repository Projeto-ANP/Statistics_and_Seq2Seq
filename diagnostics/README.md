# Diagnostics — instrumentação do orchestrator ReAct

Pasta única para todo o ciclo de diagnóstico do problema "agente escolhe pior
que as sementes". Nenhum script aqui chama LLM na máquina local: os que rodam
no servidor (`diag_entry.py`) apenas DISPARAM o run; toda a análise
(`analyze_run.py`) é replay determinístico.

## O que cada arquivo faz

| arquivo | onde roda | o que faz |
|---|---|---|
| `diag_entry.py` | servidor (precisa do Ollama) | driver de um braço: dataset + versão + seed + `--reasoning` + overrides de `ReactConfig` via `--config` + `--use-llm 0` para o braço determinístico. Escreve CSV + artifacts em `timeseries/mestrado/resultados/orchestrator_react_<version>/` |
| `run_diagnostics_server.sh` | servidor | batelada completa (~2,5 h): âncoras (determinístico + controle), 3 seeds, `reasoning off/low`, `calibration_gate`, `no-card`, `no-seed-pooled` em NN5 + ETTM2. Log de cada braço em `diagnostics/logs/<braço>_<dataset>.log` |
| `analyze_run.py` | **local** (sem LLM) | analisa a pasta trazida do servidor: final vs melhor-semente-no-teste por série, origem do vencedor, transferência das propostas (validação→teste), Spearman(val, teste), verdicts, comparação com o braço determinístico |

## Protocolo de uma rodada de diagnóstico

1. **Servidor** (a pasta precisa existir antes do redirect — o `>` abre o arquivo
   antes de o script executar o `mkdir`):
   ```bash
   mkdir -p diagnostics/logs
   nohup bash diagnostics/run_diagnostics_server.sh > diagnostics/logs/_batch.log 2>&1 &
   ```
2. Trazer de volta: as pastas `timeseries/mestrado/resultados/orchestrator_react_diag_*`
   (CSV + `llm_artifacts/`) e `diagnostics/logs/`.
3. **Local** (env `agno`):
   ```
   python diagnostics/analyze_run.py --run orchestrator_react_diag_control_nn5 \
       --dataset NN5_WEEKLY_DATASET --tsf nn5_weekly_dataset.tsf \
       --det orchestrator_react_diag_det_nn5
   ```

## Armadilhas conhecidas (leia antes de desenhar novos braços)

- **`--indices` < 20 séries muda a arquitetura do agente** (pooled meta-model
  e DATASET CARD somem). Não use runs parciais como braço de ablação; rode o
  dataset inteiro e filtre na análise com `--series`.
- Cada braço precisa da **âncora de controle** (`diag_control`, seed 7 =
  config publicada) para separar efeito da flag de variância de amostragem.
- Reporte **por série**, não só a média do dataset (n=7 no ETTM2 é dominado
  por 1-2 séries).

## Evidência que motivou esta batelada

Ver `ENTENDIMENTO_ORCHESTRATOR_REACT.md` §2-§5: NN5 determinístico 0.1157 <
agente 0.1177; oráculo das sementes 0.1060; transferência das propostas do
agente 9/93 (10%) no NN5 vs 5/5 (100%) no ETTM2; verdict `indistinguishable`
em 99/111 séries do NN5.
