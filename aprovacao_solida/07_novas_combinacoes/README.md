# 07 — Novas combinações testadas offline (antes de entrarem no catálogo)

Testa, de forma determinística e só com validação, as combinações novas que
podem entrar no catálogo do agente:

1. `best_per_horizon`: pesos one-hot por passo do horizonte, escolhendo o melhor
   preditor em cada passo pela validação (existia no orquestrador antigo e sumiu
   no ReAct).
2. `topk_mean_per_horizon`: média dos k melhores preditores por passo.
3. `forward_selection`: pesos de Caruana (seleção gulosa com reposição),
   clássica da literatura de seleção de ensembles.
4. `forward_mean`: subconjunto de Caruana combinado por média.
5. `best_single_validation`: melhor modelo individual escolhido na validação.

Cada candidata é comparada ao piso (melhor das sementes) em duas leituras: se
ela teria sido selecionada pela pontuação de validação, e qual é o resultado
dela na janela de teste. Nada aqui altera o catálogo, o prompt ou as sementes: as
receitas novas vivem em `orchestrator_react/weighting.py` e `selection.py` e só
entram no catálogo (item 1 do plano) se passarem neste teste.

## Comandos

```bash
conda activate agno
cd ~/Statistics_and_Seq2Seq

# Smoke: cinco séries
python3 aprovacao_solida/07_novas_combinacoes/run_new_combos.py \
    --datasets ANP_MONTHLY --limit 5

# Completo, todos os datasets
nohup python3 aprovacao_solida/07_novas_combinacoes/run_new_combos.py \
    > logs/new_combos.log 2>&1 &

# Com os pesos de pontuação escolhidos por validação (auto_loo)
python3 aprovacao_solida/07_novas_combinacoes/run_new_combos.py \
    --score-preset auto_loo
```

## Saída

`outputs/resultados/aprovacao_solida/novas_combinacoes.csv`, com uma linha por
(série, método): score de validação, `beats_floor_validation`, SMAPE, POCID e
RMSE de teste, e a especificação. O terminal resume, por dataset, média de
SMAPE/POCID, quantas séries cada candidata ganha do piso no teste e em quantas
ela seria selecionada pela validação.

## Como decidir

Uma candidata entra no catálogo somente se: (a) for selecionada pela validação
com frequência razoável, e (b) confirmar a vantagem no teste de forma
consistente entre datasets. Se a validação seleciona e o teste não confirma, é
caso de winner's curse e a candidata fica de fora. Se a validação não seleciona
mas o teste ganha, vale entender por quê antes de adotar (pode ser ruído de
poucas séries).

## Ligação com o resto

- As receitas já estão no core atrás dos métodos `best_per_horizon`,
  `topk_mean_per_horizon` e `forward_selection` (`weighting.py`) e `forward`
  (`selection.py`), com testes em `tests/test_new_recipes.py`.
- O item 1 do plano (consolidação do catálogo e prompt v2) acrescenta as
  candidatas aprovadas como ações e/ou sementes.
- O `run_all.py` inclui este passo como `07_novas_combinacoes` (etapa "combos").
