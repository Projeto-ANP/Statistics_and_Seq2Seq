# 03 — Estatística pareada por dataset (com Holm)

**Pergunta**: as comparações do capítulo resistem quando feitas **dentro de cada
dataset** (pareadas por série) e com correção de multiplicidade? A versão atual
junta 680 séries de 7 datasets num único Wilcoxon; esta é a versão que a banca
pede.

## Comandos (copiar e colar no servidor)

```bash
conda activate agno
cd ~/Statistics_and_Seq2Seq

# Sem os braços de busca (só runs existentes)
python3 aprovacao_solida/03_estatistica_por_dataset/stats_por_dataset.py

# Incluindo os braços do item 01 (rode depois de terminar o 01)
python3 aprovacao_solida/03_estatistica_por_dataset/stats_por_dataset.py \
    --search-version v1

# Referência diferente (default: sem_agente); exemplo com CREST:
python3 aprovacao_solida/03_estatistica_por_dataset/stats_por_dataset.py \
    --reference CREST
```

## O que sai

* Tabela de médias por dataset/metodo (SMAPE, POCID, RMSE) no terminal.
* Lista das comparações significativas com `p` e `p_holm`, W/T/L e a direção.
* `outputs/resultados/aprovacao_solida/stats_por_dataset.csv` — todas as
  comparações (uma linha por dataset × metodo × metrica).

## Como usar no texto

* Reporte a tabela por dataset (médias) e, em nota, as comparações que
  sobrevivem a Holm. O que não sobrevive vira "empate dentro do ruído".
* A frase que a banca quer ler: "as diferenças vs ADE sobrevivem à correção
  quando presentes; vs FFORMA e mediana, a comparação é estatisticamente um
  empate". (Confirme com os números que saírem; o script mostra a direção.)
* `pocid` é maior-melhor; o script já cuida da direção em W/T/L e na marcação
  "melhor/pior".
