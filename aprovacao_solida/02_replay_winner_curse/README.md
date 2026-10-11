# 02 — Replay offline: vencedor vs melhor semente (winner's curse)

**Pergunta**: o vencedor que os runs aplicaram realmente ganha da melhor semente
**na janela de teste**? E se a decisão final fosse um *gate* (só trocar a semente
pelo agente quando ele melhora o score de validação por X%), quanto mudaria?

Não roda LLM: relê o CSV do run, reconstrói as sementes com o mesmo pré-passo
determinístico, reaplica as duas estratégias ao teste e compara. A decisão de
cada regra usa **apenas validação**; o teste só mede o resultado.

Por que importa: a estratégia final é o mínimo de ~10 sementes + até 12
tentativas do agente, todas pontuadas em 3 janelas. Quanto mais tentativas, mais
o "melhor da validação" se ajusta ao ruído — e pode perder no teste para uma
semente que já estava lá. Este script mede isso e testa a correção.

## Comandos (copiar e colar no servidor)

```bash
conda activate agno
cd ~/Statistics_and_Seq2Seq
mkdir -p logs

# 1) SMOKE: 3 series do run v5 (~2 min)
python3 aprovacao_solida/02_replay_winner_curse/replay_gate.py \
    --run orchestrator_react_v5 --datasets ANP_MONTHLY --limit 3 --no-artifacts

# 2) FULL: um run inteiro (gpt-oss), todos os datasets. ~1-3 h.
nohup python3 aprovacao_solida/02_replay_winner_curse/replay_gate.py \
    --run orchestrator_react_v5 \
    > logs/replay_gate_v5.log 2>&1 &

# 3) Idem para a variante qwen, se quiser confirmar em dois modelos
nohup python3 aprovacao_solida/02_replay_winner_curse/replay_gate.py \
    --run orchestrator_react_v5_qwen \
    > logs/replay_gate_qwen.log 2>&1 &
```

Sem `--no-artifacts`, o script tenta ler o veredito `separated`/`indistinguishable`
dos `llm_artifacts_path` de cada série (só funciona no servidor, onde os
artefatos existem). Se o caminho não existir, a regra `verdict_separated` cai
para a semente (conservadora).

## O que sai

* `outputs/resultados/aprovacao_solida/replay_gate_<run>.csv` — por série:
  scores de validação (vencedor e semente), `same_forecast`, SMAPE/POCID do
  vencedor e da semente, e o pick/SMAPE de cada regra.
* `..._por_dataset.csv` — médias por dataset (a tabela que vai para o texto).
* No log: por dataset, o resumo com `seed_beats_winner_test`,
  `winner_beats_seed_test` e o SMAPE médio de cada regra.

## Como ler

* Se `seed_beats_winner_test` for da ordem de 30–40% das séries, o winner's
  curse é real e documentável — e vira um parágrafo de resultados, não um
  problema escondido.
* Compare `argmin_smape` (hoje) com `margin_0.01_smape` / `margin_0.02_smape`:
  se o gate tiver SMAPE médio menor, rode a versão final com
  `--final-strategy seed_gate --seed-gate-margin 0.01` (item 06).
* `same_forecast=True` significa que o vencedor e a semente produzem a mesma
  previsão; o gate é um no-op nesses casos.

Tempo: ~1–3 h por run completo (3 s por série no ANP, menos nos ETT). Pode rodar
por dataset com `--datasets X` para fatiar.
