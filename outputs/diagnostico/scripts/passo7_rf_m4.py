"""Passo 7 -- a divergencia do RF no M4 e o mesmo mecanismo do CatBoost
(retroalimentacao recursiva instavel) ou o problema classico do denominador
pequeno do MAPE?

Casos: toda linha de outputs/resultados/diagnostico/passo5_contexto_divergencia.csv
com dataset=M4_WEEKLY_DATASET e regressor na familia RF.

Para cada caso:
- test/predictions completos (Passo 1 do prompt).
- log10(max|pred| / max|test|) -- mesma metrica do passo1_exemplos_brutos.json.
- ponto do horizonte com maior APE pontual (|pred_i-test_i|/|test_i|): test
  nesse ponto e "pequeno" (<1 ou <1% da media de |test| da serie) e a previsao
  nesse ponto e "comparavel" as demais previsoes da mesma serie (nao mais que
  10x a mediana de |pred| dos outros pontos)? (Passo 2 do prompt.)
- y_train reconstruido do .tsf original (train = serie completa menos os
  ultimos `horizon` pontos -- alinhamento com `test` verificado manualmente
  antes de escrever este script: bate exatamente para os 4 dataset_index
  envolvidos).

  O limite teorico do bagging (Passo 3) NAO e testado contra o intervalo
  bruto [min(y_train), max(y_train)]: o modelo nunca ve y_train na escala
  bruta. rolling_window (all_functions.py:190-209) normaliza cada linha de
  treino pela media/desvio-padrao DAQUELA JANELA especifica antes de virar
  alvo de treino, e reverse_regressors (all_functions.py:1060-1066) so
  desnormaliza a predicao final com UM mean/std fixo (janela final de
  y_train), diferente do usado em cada linha de treino. O teste correto e:
  (1) reconstruir a predicao bruta do modelo revertendo essa desnormalizacao
  final; (2) reconstruir a distribuicao dos alvos normalizados que o modelo
  realmente viu no treino (reaplicando a mesma normalizacao por janela de
  rolling_window a y_train); (3) checar se a predicao bruta cai dentro do
  min/max dessa distribuicao de alvos de treino -- o limite real do bagging.

  IMPORTANTE: essa reconstrucao (passo (2) acima) so e valida para o
  regressor "rf" puro, que treina via run_tsf_normal_series ->
  rolling_window (all_functions.py:190-209). As variantes com wavelet/
  Fourier (CWT_rf, DWT_rf, FT_rf, ONLY_*_rf) treinam via
  run_tsf_image_series -> rolling_window_image (run_tsf_regressors.py:411),
  que normaliza o alvo com znorm_by (all_functions.py:85-91) -- uma funcao
  DIFERENTE, com guarda de epsilon (abs(std) < NORM_EPS -> retorna 0.0) que
  rolling_window nao tem. Testar as 6 variantes com wavelet contra a
  distribuicao de rolling_window seria testar o limite errado. Por isso a
  coluna `limite_bagging_testavel` so e True para "rf"; nas outras 6, a
  coluna `predictions_dentro_do_limite_bagging` fica None (nao testado).

Saidas:
  outputs/diagnostico/passo7_rf_m4_casos.json (arrays completos, formato do passo1)
  outputs/diagnostico/passo7_rf_m4_classificacao.csv
"""
import json
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag_lib as D

import numpy as np
import pandas as pd

DATASET = "M4_WEEKLY_DATASET"
HORIZON = 13  # @horizon no m4_weekly_dataset.tsf
RF_FAMILY = D.RF_FAMILY
PASSO5_PATH = os.path.join(D.REPO_ROOT, "outputs", "resultados", "diagnostico", "passo5_contexto_divergencia.csv")

LOG10_RATIO_INSTABILIDADE = 10.0  # "dezenas de ordens de grandeza"
TEST_PEQUENO_ABS = 1.0
TEST_PEQUENO_REL_MEDIA = 0.01
PRED_COMPARAVEL_MULT = 10.0


def rolling_window_normalized_targets(y_train, window):
    """Reproduz all_functions.py:190-209 (rolling_window) para reconstruir a
    distribuicao dos alvos normalizados que o modelo viu no treino."""
    targets = []
    for i in range(len(y_train) - window):
        w = np.array(y_train[i:i + window])
        target = y_train[i + window]
        std = np.std(w)
        mean = np.mean(w)
        targets.append((target - mean) / std if std > 0 else target - mean)
    return targets


def rf_divergent_m4_cases():
    df = pd.read_csv(PASSO5_PATH)
    df = df[(df["dataset"] == DATASET) & (df["regressor"].isin(RF_FAMILY))]
    return list(df[["regressor", "dataset_index"]].itertuples(index=False, name=None))


def main():
    cases = rf_divergent_m4_cases()
    series_values = D.tsf_series_values(DATASET)

    raw_records = []
    class_rows = []

    for regressor, dataset_index in cases:
        df = D.C.load_filtered(regressor, DATASET)
        row = df[df["dataset_index"] == dataset_index].iloc[0]
        test = D.C.parse_num_list(row["test"])
        pred = D.C.parse_num_list(row["predictions"])
        n = min(len(test), len(pred))
        test, pred = test[:n], pred[:n]

        max_test = max(abs(v) for v in test) if test else 0.0
        max_pred = max(abs(v) for v in pred) if pred else 0.0
        log10_ratio = math.log10(max_pred / max_test) if max_test > 0 and max_pred > 0 else None

        raw_records.append({
            "dataset": DATASET,
            "regressor": regressor,
            "dataset_index": int(dataset_index),
            "test": test,
            "predictions": pred,
            "max_abs_test": max_test,
            "max_abs_predictions": max_pred,
            "log10_razao_pred_sobre_test": log10_ratio,
        })

        # -- ponto de APE extrema --
        ape = []
        for t, p in zip(test, pred):
            ape.append(abs(p - t) / abs(t) if t != 0 else math.inf)
        i_star = max(range(len(ape)), key=lambda i: ape[i])
        test_istar = test[i_star]
        pred_istar = pred[i_star]
        mean_abs_test = sum(abs(v) for v in test) / len(test)

        outros_pred_abs = [abs(p) for j, p in enumerate(pred) if j != i_star]
        mediana_outros_pred = statistics.median(outros_pred_abs) if outros_pred_abs else None

        test_pequeno = abs(test_istar) < TEST_PEQUENO_ABS or (
            mean_abs_test > 0 and abs(test_istar) < TEST_PEQUENO_REL_MEDIA * mean_abs_test
        )
        pred_comparavel = (
            mediana_outros_pred is not None
            and mediana_outros_pred > 0
            and abs(pred_istar) <= PRED_COMPARAVEL_MULT * mediana_outros_pred
        )

        # -- limite teorico do bagging (na escala que o modelo realmente viu) --
        # so valido para "rf" puro (rolling_window) -- ver nota no docstring
        # do modulo sobre por que as 6 variantes com wavelet nao usam este teste.
        full_series = series_values[dataset_index]
        y_train = full_series[:-HORIZON]
        window = HORIZON  # run_tsf_regressors.py: window = horizon
        limite_testavel = regressor == "rf"

        if limite_testavel:
            mean_final = float(np.mean(y_train[-window:]))
            std_final = float(np.std(y_train[-window:])) or 1e-12
            raw_model_outputs = [(p - mean_final) / std_final for p in pred]

            norm_targets_treino = rolling_window_normalized_targets(y_train, window)
            min_norm_treino = min(norm_targets_treino)
            max_norm_treino = max(norm_targets_treino)
            max_abs_norm_treino = max(abs(min_norm_treino), abs(max_norm_treino))
            n_janelas_extremas = sum(1 for v in norm_targets_treino if abs(v) > 1e6)

            dentro_do_limite = all(min_norm_treino <= r <= max_norm_treino for r in raw_model_outputs)
            pontos_fora = [
                {"horizon_step": j + 1, "pred_bruto_normalizado": r}
                for j, r in enumerate(raw_model_outputs)
                if not (min_norm_treino <= r <= max_norm_treino)
            ]
        else:
            min_norm_treino = max_norm_treino = max_abs_norm_treino = None
            n_janelas_extremas = None
            dentro_do_limite = None
            pontos_fora = []

        if log10_ratio is not None and log10_ratio > LOG10_RATIO_INSTABILIDADE:
            mecanismo = "retroalimentacao_instavel"
        elif test_pequeno and pred_comparavel:
            mecanismo = "denominador_pequeno"
        else:
            mecanismo = "ambiguo"

        class_rows.append({
            "regressor": regressor,
            "dataset_index": int(dataset_index),
            "max_abs_test": max_test,
            "max_abs_predictions": max_pred,
            "log10_razao_pred_sobre_test": log10_ratio,
            "horizon_step_ape_max": i_star + 1,
            "ape_max": ape[i_star] if math.isfinite(ape[i_star]) else None,
            "test_no_ponto_ape_max": test_istar,
            "pred_no_ponto_ape_max": pred_istar,
            "test_pequeno_no_ponto": test_pequeno,
            "pred_comparavel_as_demais": pred_comparavel,
            "limite_bagging_testavel": limite_testavel,
            "min_alvo_normalizado_treino": min_norm_treino,
            "max_alvo_normalizado_treino": max_norm_treino,
            "max_abs_alvo_normalizado_treino": max_abs_norm_treino,
            "n_janelas_treino_com_alvo_normalizado_acima_1e6": n_janelas_extremas,
            "predictions_dentro_do_limite_bagging": dentro_do_limite,
            "n_pontos_fora_do_limite": len(pontos_fora),
            "mecanismo": mecanismo,
        })

    out_dir = D.DIAGNOSTICO_BASE
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, "passo7_rf_m4_casos.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(raw_records, fh, ensure_ascii=False, indent=2)

    csv_path = os.path.join(out_dir, "passo7_rf_m4_classificacao.csv")
    result = pd.DataFrame(class_rows)
    result.to_csv(csv_path, index=False)

    print(result.to_string(index=False))
    print(f"\nsalvo em {json_path} e {csv_path} ({len(result)} casos)")


if __name__ == "__main__":
    main()
