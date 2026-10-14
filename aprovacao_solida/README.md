# Aprovação sólida — pacote de experimentos para a defesa

Este pacote existe para fechar o furo metodológico central do trabalho e deixar a
dissertação defensável sem reescrever a arquitetura:

> o agente LLM **não** melhora a acurácia sobre o núcleo determinístico. Falta
> demonstrar que o problema não é "a busca" em geral — e sim o LLM — e mostrar
> qual é a contribuição real (framework anti-vazamento + meta-learning entre
> séries + rastreabilidade).

Tudo aqui é **determinístico** (roda em minutos, sem GPU) exceto o item 06, que
é opcional e só documenta como rodar novamente com `--reasoning low`.

Branch: `v1_proposta_final` (contém este pacote + dois patches no core:
`final_strategy="seed_gate"` e o salvamento do texto de reasoning nos artefatos).

## Execução única no servidor

Tudo que é determinístico foi empacotado em um único ponto de entrada. No
servidor:

```bash
conda activate agno
cd ~/Statistics_and_Seq2Seq
python3 aprovacao_solida/run_all.py
```

Isso roda, em sequência, a busca não-LLM nos sete datasets e a análise, o replay
vencedor versus semente, a mediana nas sementes, a sensibilidade da seleção aos
pesos, a estatística por dataset e a latência. Tudo fica em
`outputs/resultados/aprovacao_solida/`, que é a pasta a commitar, e os logs ficam
em `logs/aprovacao_solida/`. Para um teste rápido antes do run completo, use
`--limit 3 --datasets ETTH1`. Para ver os comandos sem executar, use
`--dry-run`.

## Mapa dos experimentos

| # | Pasta | Responde | Roda LLM? | Tempo (7 datasets) |
|---|-------|----------|-----------|--------------------|
| 01 | `01_busca_nao_llm/` | Busca simples (gulosa/aleatória/bandit) empata ou vence o agente? (isola o LLM) | não | ~10–30 min |
| 02 | `02_replay_winner_curse/` | O vencedor ganha da melhor semente no teste? Um gate na decisão final melhora? | não | ~1–3 h (por run) |
| 03 | `03_estatistica_por_dataset/` | Comparações pareadas por dataset com correção de Holm | não | ~1 min |
| 04 | `04_latencia_custo/` | Custo/latência por chamada, reasoning, tokens | não | ~1 min |
| 05 | `05_sementes_medianas/` | "A mediana ganha em tudo": adicionar mediana às pools estáveis melhora o piso? | não | ~10–30 min |
| 06 | `06_sensibilidade_score/` | A escolha dos pesos (0,3, 0,3, 0,2, 0,2) dirige os resultados? | não | minutos |
| 07 | `07_novas_combinacoes/` | As novas combinações (por horizonte, Caruana, best_single) melhoram o piso? Aprovadas, entram no catálogo | não | ~30 min |
| 08 | `08_reasoning_e_harness/` | Como (e se) rerodar com `--reasoning low`, mais iterações e `seed_gate` | sim | horas |

No servidor, toda a bateria determinística roda em um único comando:
`python3 aprovacao_solida/run_all.py` (Seção "Execução única no servidor").
A rerodada com LLM, item 08, só depois, e só se os itens anteriores pedirem.

## Respostas diretas às suas dúvidas

1. **Dedup do NN5**: não é prioridade. A banca não conhece o problema, e o
   efeito aparece só no meta-modelo pooled do NN5 (as séries duplicadas T1≡T47,
   T11≡T50, T79≡T111 compartilham fold no LOSO). Se um membro perguntar, a
   resposta honesta é "o pooled tem um viés otimista no NN5; deduplicar é
   trabalho futuro". Não gaste tempo nisso agora.
2. **Rerodar tudo com `--reasoning low` para latência?** Não rerode às cegas.
   Faça primeiro o A/B do item 06 num subconjunto (`--limit 20`, 1 dataset).
   Se o resultado do agente for igual (esperado — ele não agrega acurácia) e a
   latência cair, aí sim rode a versão final com `--reasoning low`, de
   preferência junto com `--final-strategy seed_gate` (protege do winner's
   curse) e `--max-iterations 20`. **Nunca misture configurações entre braços
   comparados.**
3. **"O agente pode perder a melhor por tentar outra"**: no papel ele não pode —
   a estratégia aplicada é sempre o mínimo de todo o histórico. O que acontece é
   outra coisa, e é real: quanto mais tentativas, mais o *máximo do histórico*
   se ajusta ao ruído das 3 janelas (winner's curse), e aí o vencedor pode ser
   pior no teste que a semente. O item 02 mede exatamente isso e o `seed_gate`
   do item 06 é a correção barata.
4. **Mediana sempre ganha**: o item 05 testa, no piso determinístico, incluir
   mediana nas pools estáveis (hoje só há média/média aparada). É a mudança de
   melhor custo/benefício se confirmar no teste.

## O que cada resultado significa para a defesa

* **01** — se `search_best` (melhor braço escolhido na validação) empata ou
  vence o CREST no teste: conclusão "LLM não agrega, e busca simples também não"
  → o valor está no meta-learning/semeadura. Se o CREST vence a busca: a decisão
  do agente tem mérito próprio (raro, mas possível onde a evidência é escassa).
* **02** — quantas vezes a semente é melhor que o vencedor no teste; se uma
  regra de gate (ex.: `margin_0.01`) tem SMAPE médio menor que `argmin`, vale
  rerodar com `seed_gate`.
* **03** — a tabela que a banca vai pedir: por dataset, pareada, com Holm.
* **04** — custo-benefício: "o agente custa X min/série e não melhora SMAPE".
* **05** — evidência para (ou contra) a inclusão da mediana no piso.
* **06** — protocolo de rerodada, se e somente se os itens acima pedirem.

## Comandos rápidos (smoke, ~2 min total)

```bash
conda activate agno
cd ~/Statistics_and_Seq2Seq

python3 aprovacao_solida/01_busca_nao_llm/run_search.py \
    --dataset ANP_MONTHLY --indices 0 1 2 --version smoke \
    --methods seeds,random,greedy,ucb --no-pooled-meta-model --quiet

python3 aprovacao_solida/02_replay_winner_curse/replay_gate.py \
    --run orchestrator_react_v5 --datasets ANP_MONTHLY --limit 3 --no-artifacts

python3 aprovacao_solida/04_latencia_custo/analyze_latency.py \
    --versions orchestrator_react_v5 --datasets ANP_MONTHLY

python3 aprovacao_solida/05_sementes_medianas/seed_medians.py \
    --datasets ANP_MONTHLY --limit 3 --no-pooled-meta-model
```

## Regras gerais

* Rode a partir da raiz do repositório (`cd ~/Statistics_and_Seq2Seq`).
* Sempre `conda activate agno` antes (tem `aeon`, `statsmodels`, `xgboost`).
* `mkdir -p logs` antes dos `nohup`.
* `--no-pooled-meta-model` só para smoke/CPU fraca; no servidor deixe o pooled
  ligado (é a décima semente e a comparação precisa dela).
* Nenhum script desta pasta escreve nos CSVs dos runs existentes; todos gravam
  em pastas novas (`orchestrator_search_*`, `outputs/resultados/aprovacao_solida/`).
