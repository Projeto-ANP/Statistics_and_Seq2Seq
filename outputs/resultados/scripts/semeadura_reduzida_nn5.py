"""reduced_seeding no NN5 (111 series), SEM agente: piso deterministico (melhor semente).

Compara duas execucoes de run_tsf_baselines.py com o mesmo codigo, uma com as 10 sementes
e outra com --reduced-seeding (8). Nao mede o efeito com o agente (isso exige LLM).

Uso: python3 semeadura_reduzida_nn5.py <csv_controle> <csv_reduzido>
"""
import json, sys
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

EPS = 1e-4


def load(p):
    df = pd.read_csv(p, sep=";", engine="python")
    df["final_test"] = pd.to_datetime(df["final_test"])
    df = df.sort_values("final_test", ascending=False).drop_duplicates(["dataset_index", "regressor"]).sort_values("dataset_index")
    return df.set_index("dataset_index")


def strat(r):
    d = json.loads(r["best_strategy_params"])
    return json.dumps(d, sort_keys=True), d


a, b = load(sys.argv[1]), load(sys.argv[2])
idx = a.index.intersection(b.index)
print(f"series: {len(idx)}")
for m in ("smape", "pocid", "rmse"):
    x, y = a.loc[idx, m].astype(float), b.loc[idx, m].astype(float)
    print(f"{m}: controle {x.mean():.5f} | reduzida {y.mean():.5f} | dif {y.mean()-x.mean():+.5f}")
    d = (y - x).values
    if m == "smape":
        print(f"  reduzida melhor (smape menor) em {(d < -EPS).sum()}, pior em {(d > EPS).sum()}, empate em {(np.abs(d) <= EPS).sum()}")
        nz = d[np.abs(d) > 0]
        if len(nz) >= 5:
            print(f"  Wilcoxon (series com dif != 0, n={len(nz)}): p={wilcoxon(nz).pvalue:.4f}")
    if m == "pocid":
        print(f"  reduzida maior pocid em {(d > 1e-9).sum()}, menor em {(d < -1e-9).sum()}, igual em {(np.abs(d) <= 1e-9).sum()}")
sa = {i: strat(a.loc[i]) for i in idx}
sb = {i: strat(b.loc[i]) for i in idx}
same = sum(sa[i][0] == sb[i][0] for i in idx)
print(f"estrategia final identica: {same} de {len(idx)}")
lost = [i for i in idx if sa[i][0] != sb[i][0]]
was_full = [i for i in lost if sa[i][1].get("combine") in ("mean", "median") and sa[i][1].get("pool") == "pool_full"]
print(f"series com estrategia final diferente: {len(lost)}; delas, controle tinha mean/median do pool completo: {len(was_full)}")
print(f"no controle, mean/median do pool completo vencia em {sum(1 for i in idx if sa[i][1].get('combine') in ('mean','median') and sa[i][1].get('pool')=='pool_full')} series")
