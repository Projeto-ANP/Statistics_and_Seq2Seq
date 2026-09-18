"""Item 15 -- Candidatos a estudo de caso.

Para orchestrator_react_v5 (todos os datasets juntos): ate 10 series com maior
n_trajectory_steps cuja estrategia final tem origin="baseline", e,
separadamente, ate 10 series com maior n_trajectory_steps cuja estrategia
final tem origin="agent". Para cada serie: dataset, dataset_index,
n_trajectory_steps, react_trajectory_json e justificativa_final completos.

Saida: outputs/resultados/item_15/candidatos_estudo_caso.json
  chaves: "origin_baseline", "origin_agent" (nome literal do campo/valor de
  origem que discrimina os dois grupos, description_json.origin).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

TOP_N = 10


def main():
    warnings = []
    records = []

    for dataset in C.DATASETS:
        df = C.load_orchestrator(C.ORCHESTRATOR_GPTOSS, dataset)
        warning = C.check_series_count(dataset, C.canonical_regressor_name(C.ORCHESTRATOR_GPTOSS), len(df))
        if warning:
            warnings.append(warning)

        for row in df.itertuples(index=False):
            desc = row.description_json
            records.append({
                "dataset": dataset,
                "dataset_index": int(row.dataset_index),
                "origin": desc.get("origin"),
                "n_trajectory_steps": C.dget(desc, "loop", "n_trajectory_steps"),
                "react_trajectory_json": row.react_trajectory_json_parsed,
                "justificativa_final": row.justificativa_final,
            })

    def top_n_for(origin_value):
        candidates = [r for r in records if r["origin"] == origin_value and r["n_trajectory_steps"] is not None]
        candidates.sort(key=lambda r: r["n_trajectory_steps"], reverse=True)
        out = []
        for r in candidates[:TOP_N]:
            out.append({k: v for k, v in r.items() if k != "origin"})
        return out

    result = {
        "origin_baseline": top_n_for("baseline"),
        "origin_agent": top_n_for("agent"),
    }

    out_dir = os.path.join(C.OUTPUTS_BASE, "item_15")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "candidatos_estudo_caso.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)

    C.merge_validation_warnings(warnings)
    print(
        f"salvo em {out_path} "
        f"({len(result['origin_baseline'])} origin_baseline, {len(result['origin_agent'])} origin_agent)"
    )


if __name__ == "__main__":
    main()
