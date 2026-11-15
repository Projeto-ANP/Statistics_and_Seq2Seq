# Ablação do CARTE: buscas sem LLM e oráculos

## Comando no servidor

```bash
cd ~/Statistics_and_Seq2Seq
conda activate agno
mkdir -p logs
nohup bash outputs/resultados/scripts/run_ablacao_carte.sh \
    > logs/ablacao_carte_master.log 2>&1 &
```

Acompanhar:

```bash
tail -f logs/ablacao_carte_master.log
```

## O que é executado

1. `ablacao_search_carte.py` — busca aleatória (sementes 7, 8 e 9), gulosa e UCB, com 12 avaliações por série, histórico inicial vazio, mesmo catálogo e mesma regra final do CARTE.
2. `ablacao_oraculos_carte.py` — melhor tentativa, melhores dez primeiras tentativas, teto de validação sobre as tentativas e melhor modelo individual, sempre com a média das dez execuções.
3. `ablacao_oraculo_u_carte.py` — oráculo amplo sobre o universo fixo de estratégias, com o teto de validação e o oráculo de teste. Grava incrementalmente e aceita `--resume`.
4. `ablacao_oraculos_resumo.py` — consolida os oráculos em uma tabela única.
5. `ablacao_search_resumo.py` — agrega as buscas por conjunto de dados e calcula os testes pareados.

## Saídas

- `timeseries/mestrado/resultados/orchestrator_search_carte/*.csv`
- `outputs/resultados/ablacao_search_carte/artifacts/`
- `outputs/resultados/ablacao_oraculos_carte/`
- `outputs/resultados/ablacao_oraculo_u_carte/`
- `outputs/resultados/ablacao_oraculos_resumo/`
- `outputs/resultados/ablacao_search_carte_resumo/`

## Retomada

O comando pode ser repetido. A busca usa `--resume` e não refaz pares (série, braço) já gravados; o oráculo amplo também usa `--resume` e não refaz séries já presentes em `por_serie.csv`.
