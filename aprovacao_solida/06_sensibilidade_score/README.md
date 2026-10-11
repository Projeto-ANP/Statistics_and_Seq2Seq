# 06 — Sensibilidade da seleção aos pesos da pontuação

Responde à pergunta do orientador sobre a escolha de (0,3, 0,3, 0,2, 0,2) para
RMSE, SMAPE, MAPE e POCID. Para cada série, as mesmas estratégias candidatas
(as dez sementes recalculadas e o vencedor aplicado pelo run) são repontuadas
sob cada preset de `orchestrator_react/config.py` (`balanced`, `rmse_focus`,
`direction_focus`, `robust_smape`, `scale_free_safe`), o argmin de cada preset
é tomado, e dois efeitos são medidos: em quantas séries a estratégia
selecionada muda, e quanto o SMAPE e o POCID de teste da estratégia
selecionada variam entre presets.

Tudo determinístico, sem LLM.

## Comando

```bash
conda activate agno
cd ~/Statistics_and_Seq2Seq
python3 aprovacao_solida/06_sensibilidade_score/run_score_sensitivity.py \
    --run orchestrator_react_v5
```

Smoke:

```bash
python3 aprovacao_solida/06_sensibilidade_score/run_score_sensitivity.py \
    --run orchestrator_react_v5 --datasets ANP_MONTHLY --limit 20
```

## Saída

`outputs/resultados/aprovacao_solida/sensibilidade_score.csv`, com uma linha
por série e, para cada preset, a estratégia selecionada, seu score de validação
e o SMAPE e o POCID de teste; mais `selection_changed_vs_balanced` e
`smape_delta_max_preset`.

## Como ler

Se a seleção muda em poucas séries e o SMAPE médio de teste varia pouco entre
presets, a escolha dos pesos não dirige os resultados e a justificativa escrita
no documento se sustenta. Se mudar muito, a resposta correta ao orientador é
que os pesos são um parâmetro de projeto com efeito mensurável, e o número de
séries afetadas deve entrar no texto.
