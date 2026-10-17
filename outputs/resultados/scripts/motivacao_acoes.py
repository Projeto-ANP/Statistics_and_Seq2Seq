#!/usr/bin/env python3
"""Explora os pensamentos (thought) das trajetorias finais para explicar por que
algumas acoes do catalogo sao chamadas muito mais que as outras.

Le os runs finais (orchestrator_react_v2_medium_x1_gpt-oss-20b, 7 datasets) e
agrega, por acao do catalogo:

- numero de chamadas (sanidade contra item_10);
- termos de conteudo mais frequentes nos pensamentos que antecedem a chamada
  (frequencia em documentos, nao so contagem de tokens);
- fracao dos pensamentos que citam motivos conhecidos do diagnostico
  (tendencia, ranking instavel, erros correlacionados, redundancia, robustez);
- para a avaliacao de estrategia, os termos mais frequentes no campo
  `rationale`, a justificativa curta que o agente escreve ao avaliar.

Saidas:
  outputs/resultados/item_10/motivacao_acoes.md
  outputs/resultados/item_10/motivacao_acoes_termos.csv
"""
from __future__ import annotations

import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

OUT_DIR = os.path.join(C.OUTPUTS_BASE, "item_10")

STOP = set(
    "a an the and or but if then else of to in on for with without at by from as is are was were "
    "be been being it its this that these those their there here they them he she his her we our "
    "you your i not no nor do does did doing have has had having can could may might must should "
    "would will shall about into over under again further once only own same so than too very just "
    "don now also more most other some such out up down between within across per each both few all "
    "any because while during before after above below off".split()
)

MOTIVOS = {
    "tendencia (trend)": r"trend",
    "sazonalidade (season)": r"season",
    "ranking instavel (unstable)": r"unstable",
    "ranking estavel (stable)": r"stable",
    "ranking": r"ranking",
    "erros correlacionados (correlat*)": r"correlat",
    "redundancia (redundan*)": r"redundan",
    "robustez (robust)": r"robust",
    "erro inverso (inverse)": r"inverse",
    "mediana (median)": r"median",
    "media simples (mean)": r"\bmean\b",
    "sobreajuste (overfit)": r"overfit",
    "validacao (validation)": r"validation",
    "janelas (window)": r"window",
    "margem (margin)": r"margin",
    "semente (seed)": r"\bseed",
    "podar (prune)": r"prun",
    "pesos (weight)": r"weight",
    "horizonte (horizon)": r"horizon",
    "erro (error)": r"error",
    "simplicidade (simpl*)": r"simpl",
    "diversidade (divers*)": r"divers",
    "shrinkage (shrink)": r"shrink",
    "meta-modelo (meta)": r"\bmeta",
    "forward": r"forward",
    "concentracao (concentr*)": r"concentr",
}


def tokens(text: str) -> list[str]:
    text = text.lower().replace("\u2011", " ").replace("-", " ")
    return [t for t in re.findall(r"[a-z']+", text) if t not in STOP and len(t) > 2]


def bigrams(toks: list[str]) -> list[str]:
    return [f"{a} {b}" for a, b in zip(toks, toks[1:])]


def coletar() -> list[dict]:
    records = []
    for ds in C.DATASETS:
        df = C.load_orchestrator(C.ORCHESTRATOR_GPTOSS, ds)
        for row in df.itertuples(index=False):
            traj = getattr(row, "react_trajectory_json_parsed", None) or []
            for step in traj:
                args = step.get("action_args") or {}
                records.append({
                    "dataset": ds,
                    "dataset_index": int(row.dataset_index),
                    "iteration": step.get("iteration"),
                    "action": str(step.get("action") or ""),
                    "thought": str(step.get("thought") or ""),
                    "rationale": str(args.get("rationale") or ""),
                })
    return records


def top_doc_terms(textos: list[str], n: int = 12) -> list[tuple]:
    cont = Counter()
    for t in textos:
        cont.update(set(tokens(t)))
    return cont.most_common(n)


def top_bigrams(textos: list[str], n: int = 10, min_count: int = 5) -> list[tuple]:
    cont = Counter()
    for t in textos:
        cont.update(set(bigrams(tokens(t))))
    return [(b, c) for b, c in cont.most_common(60) if c >= min_count][:n]


def motivar(textos: list[str]) -> dict:
    low = [t.lower() for t in textos]
    out = {}
    for label, pat in MOTIVOS.items():
        rx = re.compile(pat)
        out[label] = sum(1 for t in low if rx.search(t))
    return out


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    records = coletar()
    por_acao: dict[str, list[dict]] = {}
    for r in records:
        por_acao.setdefault(r["action"], []).append(r)

    contagem = sorted(((a, len(v)) for a, v in por_acao.items()), key=lambda x: -x[1])
    total = sum(c for _, c in contagem)
    top3 = [a for a, _ in contagem[:3]]

    linhas = []
    linhas.append("# Por que o agente escolhe cada ação (mineração dos pensamentos)\n")
    linhas.append(f"Total de chamadas observadas nas trajetórias: {total}. "
                  f"Ações por frequência: " + ", ".join(f"{a} ({c})" for a, c in contagem) + ".\n")

    linhas.append("## Motivos citados nos pensamentos, por ação\n")
    header = "| Ação | Chamadas | " + " | ".join(MOTIVOS.keys()) + " |"
    linhas.append(header)
    linhas.append("|" + "---|" * (2 + len(MOTIVOS)))
    motivos_rows = []
    for a, c in contagem:
        textos = [r["thought"] for r in por_acao[a]]
        m = motivar(textos)
        motivos_rows.append({"action": a, "chamadas": c, **m})
        cells = [f"{100*m[k]/max(1,c):.0f}%" for k in MOTIVOS]
        linhas.append(f"| {a} | {c} | " + " | ".join(cells) + " |")
    linhas.append("")

    for a in top3:
        textos = [r["thought"] for r in por_acao[a]]
        linhas.append(f"## {a} — top termos (frequência em pensamentos, n={len(textos)})\n")
        linhas.append("| termo | pensamentos | bigrama | pensamentos |")
        linhas.append("|---|---|---|---|")
        termos = top_doc_terms(textos, 12)
        bgs = top_bigrams(textos, 12)
        for i in range(max(len(termos), len(bgs))):
            t = termos[i] if i < len(termos) else ("", "")
            b = bgs[i] if i < len(bgs) else ("", "")
            linhas.append(f"| {t[0]} | {t[1]} | {b[0]} | {b[1]} |")
        linhas.append("")
        ex = sorted(por_acao[a], key=lambda r: -len(r["thought"]))[:2]
        for e in ex:
            linhas.append(f"> {e['thought'][:300]}")
            linhas.append("")

    raci = [r["rationale"] for r in por_acao.get("evaluate_strategy", []) if r["rationale"]]
    linhas.append(f"## evaluate_strategy — top termos nas justificativas curtas (n={len(raci)})\n")
    linhas.append("| termo | justificativas | bigrama | justificativas |")
    linhas.append("|---|---|---|---|")
    termos = top_doc_terms(raci, 12)
    bgs = top_bigrams(raci, 12)
    for i in range(max(len(termos), len(bgs))):
        t = termos[i] if i < len(termos) else ("", "")
        b = bgs[i] if i < len(bgs) else ("", "")
        linhas.append(f"| {t[0]} | {t[1]} | {b[0]} | {b[1]} |")
    linhas.append("")

    out_md = os.path.join(OUT_DIR, "motivacao_acoes.md")
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(linhas))

    import pandas as pd
    pd.DataFrame(motivos_rows).to_csv(os.path.join(OUT_DIR, "motivacao_acoes_termos.csv"), index=False)
    print("salvo:", out_md)
    top3_hits = {a: motivar([r["thought"] for r in por_acao[a]]) for a in top3}
    for a in top3:
        m = top3_hits[a]
        top = sorted(m.items(), key=lambda x: -x[1])[:6]
        print(f"\n{a} ({len(por_acao[a])} chamadas) — motivos: " + ", ".join(f"{k}: {v}" for k, v in top))
    print("\ntermos evaluate_strategy (rationale):")
    print(top_doc_terms(raci, 15))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
