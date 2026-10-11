# 05 — A mediana sempre ganha? Testando no piso determinístico

**Pergunta**: as pools estáveis hoje só têm média e média aparada. A mediana só
aparece no pool completo. Adicionar mediana nas pools estáveis (k=5,7,9) melhora
o **piso determinístico**? Se melhorar, é a mudança de melhor custo-benefício do
trabalho — sem LLM, sem risco de overfit, e alinhada ao que os resultados já
mostram (mediana vencendo no M4).

Determinístico, sem Ollama.

## Comandos (copiar e colar no servidor)

```bash
conda activate agno
cd ~/Statistics_and_Seq2Seq
mkdir -p logs

# 1) SMOKE (3 series)
python3 aprovacao_solida/05_sementes_medianas/seed_medians.py \
    --datasets ANP_MONTHLY --limit 3

# 2) FULL, todos os datasets (~10-30 min)
nohup python3 aprovacao_solida/05_sementes_medianas/seed_medians.py \
    > logs/seed_medians.log 2>&1 &
```

## O que sai

* Terminal, por dataset: SMAPE do piso atual vs piso com medianas, diff médio,
  contagem melhor/pior/empate e quantas vezes o piso mudou.
* `outputs/resultados/aprovacao_solida/sementes_medianas.csv` — por série, as
  duas estratégias vencedoras e as métricas de teste.

## Como agir com o resultado

* Se `com medianas` ganhar de forma consistente **no teste** (não só na
  validação), edite `orchestrator_react/pool.py`:

  ```python
  SEED_STABLE_POOLS = (
      (5, "mean"), (5, "median"), (5, "trimmed_mean"),
      (7, "mean"), (7, "median"), (7, "trimmed_mean"),
      (9, "mean"), (9, "median"), (9, "trimmed_mean"),
  )
  ```

  e rerode os braços que forem usados na versão final (o piso muda, então
  qualquer comparação com runs antigos precisa registrar a mudança).
* Se empatar, mantenha como está e cite o teste como verificação (também é
  resultado: "a mediana vence por série em vários datasets, mas não melhora o
  piso quando o subset correto é escolhido").
