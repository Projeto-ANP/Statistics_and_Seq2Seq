"""Comprimento do `thought` e custo por chamada ao LLM, por modelo (gpt-oss, qwen3, gemma4).

Fonte: CSVs dos tres orchestrator_react_v5* (colunas react_trajectory_json e
description.loop). A resposta bruta do LLM NAO e salva em nenhum artefato: so o
`thought` (limitado a 600 caracteres em react_loop.py:253), a acao, os
argumentos e o resumo da observacao. Por isso o comprimento medido aqui e o do
`thought` parseado, nao o da resposta inteira.

Saida (stdout): uma linha por modelo, formato markdown.
"""
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

MODELS = {
    "gpt-oss:20b": C.ORCHESTRATOR_GPTOSS,
    "qwen3:30b-a3b": C.ORCHESTRATOR_QWEN,
    "gemma4:26b": C.ORCHESTRATOR_GEMMA,
}
CLIP = 600


def main():
    print("| modelo | series | passos | thought (chars) media | desvio | mediana | % no limite de 600 "
          "| chamadas/serie | respostas vazias/serie | s por chamada (media) | desvio |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for name, folder in MODELS.items():
        lens, sec, calls, empty, series = [], [], [], 0, 0
        for ds in C.DATASETS:
            df = C.load_orchestrator(folder, ds)
            for r in df.itertuples(index=False):
                traj = r.react_trajectory_json_parsed
                if not traj:
                    continue
                loop = r.description_json["loop"]
                series += 1
                lens += [len(str(s.get("thought") or "")) for s in traj]
                n_calls = loop["n_trajectory_steps"] + loop.get("empty_responses", 0) + loop.get("llm_error_retries", 0)
                empty += loop.get("empty_responses", 0)
                calls.append(n_calls)
                sec.append(loop["elapsed_s"] / max(n_calls, 1))
        clipped = 100 * sum(1 for x in lens if x >= CLIP) / len(lens)
        print(f"| {name} | {series} | {len(lens)} | {st.mean(lens):.1f} | {st.pstdev(lens):.1f} | {st.median(lens):.0f} "
              f"| {clipped:.1f}% | {st.mean(calls):.1f} | {empty / series:.2f} | {st.mean(sec):.2f} | {st.pstdev(sec):.2f} |")


if __name__ == "__main__":
    main()
