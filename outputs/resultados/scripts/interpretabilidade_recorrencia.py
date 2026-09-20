"""Recorrencia de vocabulario nas justificativas do agente (origin=agent).

So mede e reporta numeros -- nenhuma interpretacao qualitativa alem do que
os Passos 1-4 do prompt pedem literalmente.

Saida: outputs/resultados/interpretabilidade/RESUMO_recorrencia_justificativas.md
"""
import itertools
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

# ---------------------------------------------------------------------------
# Tokenizacao (documentada explicitamente no RESUMO)
# ---------------------------------------------------------------------------

STOPWORDS = set("""
a an the and or but is are was were to of in on at for with this that it its
as has have had be by from which who whom these those than then so such not
no yes do does did can could will would should may might must shall i you
he she we they them their his her our your my mine ours yours theirs if
because while when where why how all each few more most other some only own
same too very s t just now also over under again further once here there
""".split())

_WORD_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list:
    """Minusculas, hifens/travessoes/pontuacao viram espaco, so [a-z0-9]+."""
    text = text.lower()
    text = re.sub(r"[‐‑‒–—-]", " ", text)
    return _WORD_RE.findall(text)


def content_words(text: str) -> set:
    return {w for w in tokenize(text) if w not in STOPWORDS}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


_CLAUSE_BOUNDARY_RE = re.compile(r",|;|\.(?!\d)")


def first_clause(text: str) -> str:
    """Ate a primeira virgula/ponto-e-virgula/ponto -- um ponto seguido de
    digito (ex.: "0.9863") nao conta como fim de frase."""
    m = _CLAUSE_BOUNDARY_RE.search(text)
    clause = text[: m.start()] if m else text
    return " ".join(clause.lower().split())


# ---------------------------------------------------------------------------
# Passo 1 -- amostra
# ---------------------------------------------------------------------------

POOL_SHAPING_TOOLS = {"prune_redundant", "select_stable", "select_top_k"}
_POOL_NAME_RE = re.compile(r"pool (\w+)")


def pool_creator_tool(row) -> str:
    """Ferramenta que criou o pool usado pela estrategia final vencedora
    (prune_redundant/select_stable/select_top_k), ou 'pool_full'/'outro'."""
    params = json.loads(row.best_strategy_params)
    target_pool = params.get("pool")
    if target_pool is None or target_pool == "pool_full":
        return "pool_full"

    creators = {}
    for step in row.react_trajectory_json_parsed:
        action = step.get("action")
        if action in POOL_SHAPING_TOOLS:
            summary = step.get("observation_summary") or ""
            m = _POOL_NAME_RE.search(summary)
            if m:
                creators[m.group(1)] = action
    return creators.get(target_pool, "outro_ou_nao_rastreado")


def collect_sample():
    records = []
    for dataset in C.DATASETS:
        df = C.load_orchestrator(C.ORCHESTRATOR_GPTOSS, dataset)
        for row in df.itertuples(index=False):
            if row.description_json.get("origin") != "agent":
                continue
            params = json.loads(row.best_strategy_params)
            estrategia_key = json.dumps(params, sort_keys=True)
            records.append({
                "dataset": dataset,
                "dataset_index": int(row.dataset_index),
                "justificativa": row.justificativa_final,
                "estrategia_params": params,
                "estrategia_key": estrategia_key,
                "pool_creator": pool_creator_tool(row),
            })
    return records


# ---------------------------------------------------------------------------
# Passo 2 -- n-gramas exatos
# ---------------------------------------------------------------------------

def ngram_doc_frequencies(justificativas: list) -> Counter:
    df_counter = Counter()
    for text in justificativas:
        words = tokenize(text)
        doc_ngrams = set()
        for n in range(5, 9):
            for i in range(len(words) - n + 1):
                doc_ngrams.add(tuple(words[i:i + n]))
        df_counter.update(doc_ngrams)
    return df_counter


OVERFIT_RE = re.compile(r"mitigat\w*\s+overfit\w*\s+(?:to\s+)?the\s+few\s+validation\s+windows")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    records = collect_sample()
    n_total = len(records)
    by_dataset = Counter(r["dataset"] for r in records)

    justificativas = [r["justificativa"] for r in records]

    # --- Passo 2 ---
    df_counter = ngram_doc_frequencies(justificativas)
    threshold = 0.20 * n_total
    frequent = [(ng, c) for ng, c in df_counter.items() if c >= threshold]
    frequent.sort(key=lambda x: (-x[1], x[0]))
    top15 = frequent[:15]

    n_overfit_phrase = sum(1 for t in justificativas if OVERFIT_RE.search(t.lower()))

    # --- Passo 3.1: Jaccard sobre todos os pares ---
    word_sets = [content_words(r["justificativa"]) for r in records]
    all_pairs_sim = []
    n = len(records)
    for i, j in itertools.combinations(range(n), 2):
        all_pairs_sim.append(jaccard(word_sets[i], word_sets[j]))
    mean_all = statistics.mean(all_pairs_sim)
    median_all = statistics.median(all_pairs_sim)

    # --- Passo 3.2: mesma estrategia vs estrategia diferente ---
    same_strategy_sim = []
    diff_strategy_sim = []
    for idx, (i, j) in enumerate(itertools.combinations(range(n), 2)):
        sim = all_pairs_sim[idx]
        if records[i]["estrategia_key"] == records[j]["estrategia_key"]:
            same_strategy_sim.append(sim)
        else:
            diff_strategy_sim.append(sim)

    # --- Passo 3.3: moldes da primeira frase ---
    first_clauses = [first_clause(r["justificativa"]) for r in records]
    exact_counts = Counter(first_clauses)
    exact_sorted = exact_counts.most_common()

    # near-duplicate clustering (Jaccard >= 0.6 on first-clause content words) via union-find
    fc_word_sets = [content_words(fc) for fc in first_clauses]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    CLUSTER_THRESHOLD = 0.6
    for i, j in itertools.combinations(range(n), 2):
        if jaccard(fc_word_sets[i], fc_word_sets[j]) >= CLUSTER_THRESHOLD:
            union(i, j)
    clusters = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(i)
    cluster_sizes = sorted((len(v) for v in clusters.values()), reverse=True)
    cum = 0
    molds_for_80 = 0
    for size in cluster_sizes:
        cum += size
        molds_for_80 += 1
        if cum >= 0.8 * n:
            break

    # --- Passo 4.1: similaridade por dataset ---
    per_dataset_stats = {}
    for dataset in C.DATASETS:
        idxs = [i for i, r in enumerate(records) if r["dataset"] == dataset]
        pairs = list(itertools.combinations(idxs, 2))
        sims = [jaccard(word_sets[i], word_sets[j]) for i, j in pairs]
        per_dataset_stats[dataset] = {
            "n_series": len(idxs),
            "n_pairs": len(pairs),
            "mean": statistics.mean(sims) if sims else None,
            "median": statistics.median(sims) if sims else None,
        }

    # --- Passo 4.2: similaridade por mecanismo de formacao do pool ---
    pool_creator_counts = Counter(r["pool_creator"] for r in records)
    per_mechanism_first_clause = defaultdict(list)
    for r, fc in zip(records, first_clauses):
        per_mechanism_first_clause[r["pool_creator"]].append(fc)

    within_mechanism_sim = {}
    for mech, idxs_text in [(m, [i for i, r in enumerate(records) if r["pool_creator"] == m]) for m in pool_creator_counts]:
        pairs = list(itertools.combinations(idxs_text, 2))
        sims = [jaccard(fc_word_sets[i], fc_word_sets[j]) for i, j in pairs]
        within_mechanism_sim[mech] = {
            "n_series": len(idxs_text),
            "n_pairs": len(pairs),
            "mean_first_clause_jaccard": statistics.mean(sims) if sims else None,
        }
    # across different mechanisms
    across_mechanism_sims = []
    mech_of = [r["pool_creator"] for r in records]
    for i, j in itertools.combinations(range(n), 2):
        if mech_of[i] != mech_of[j]:
            across_mechanism_sims.append(jaccard(fc_word_sets[i], fc_word_sets[j]))
    mean_across_mechanism = statistics.mean(across_mechanism_sims) if across_mechanism_sims else None

    # -----------------------------------------------------------------------
    # escreve RESUMO.md
    # -----------------------------------------------------------------------
    out_dir = os.path.join(C.OUTPUTS_BASE, "interpretabilidade")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "RESUMO_recorrencia_justificativas.md")

    lines = []
    lines.append("# Recorrência de vocabulário nas justificativas do agente (origin=agent)\n")
    lines.append(
        "Só números, medidos exatamente como pedido nos Passos 1-4. Sem conclusão qualitativa.\n"
    )

    lines.append("## Passo 1 — amostra\n")
    lines.append(f"Total de séries com `origin=agent`: **{n_total}** (de 680 séries no total, todos os 7 datasets).\n")
    lines.append("| dataset | n_series |")
    lines.append("|---|---|")
    for dataset in C.DATASETS:
        lines.append(f"| {dataset} | {by_dataset.get(dataset, 0)} |")
    lines.append("")

    lines.append("## Passo 2 — repetição de frases exatas (n-gramas de 5 a 8 palavras)\n")
    lines.append(
        f"Tokenização: minúsculas, hífens/travessões tratados como espaço, mantido só `[a-z0-9]+`. "
        f"N-gramas contados por presença no documento (um n-grama repetido dentro da mesma justificativa "
        f"conta uma vez). Limiar de 20% da amostra = **{threshold:.1f}** séries (arredondado para cima na contagem).\n"
    )
    lines.append(f"N-gramas de 5-8 palavras que aparecem em ≥20% das {n_total} justificativas: **{len(frequent)}**.\n")
    lines.append("Top 15 por frequência:\n")
    lines.append("| n-grama | contagem | percentual |")
    lines.append("|---|---|---|")
    for ng, c in top15:
        lines.append(f"| \"{' '.join(ng)}\" | {c} | {c/n_total*100:.1f}% |")
    lines.append("")
    lines.append(
        f"Variação de \"mitigat(ing/es) overfitting to the few validation windows\" "
        f"(regex `mitigat\\w*\\s+overfit\\w*\\s+(?:to\\s+)?the\\s+few\\s+validation\\s+windows`, "
        f"case-insensitive): aparece em **{n_overfit_phrase}/{n_total}** justificativas "
        f"(**{n_overfit_phrase/n_total*100:.1f}%**).\n"
    )

    lines.append("## Passo 3 — similaridade estrutural\n")
    lines.append(
        f"Jaccard sobre o conjunto de palavras de conteúdo (stopwords removidas), calculado sobre "
        f"**todos os {len(all_pairs_sim)} pares** da amostra (não uma subamostra aleatória — "
        f"com {n_total} séries o cálculo completo é factível e mais preciso que amostrar).\n"
    )
    lines.append(f"- Média: **{mean_all:.4f}**")
    lines.append(f"- Mediana: **{median_all:.4f}**\n")

    lines.append(
        f"Separando por `estrategia_params` (chave = `combine`+`pool`+`weights`/`trim_pct`/`model` conforme aplicável, "
        f"comparado como JSON canônico):\n"
    )
    lines.append(f"- Pares com a **mesma** estratégia (n={len(same_strategy_sim)}): "
                  f"média={statistics.mean(same_strategy_sim) if same_strategy_sim else float('nan'):.4f}, "
                  f"mediana={statistics.median(same_strategy_sim) if same_strategy_sim else float('nan'):.4f}")
    lines.append(f"- Pares com estratégia **diferente** (n={len(diff_strategy_sim)}): "
                  f"média={statistics.mean(diff_strategy_sim):.4f}, "
                  f"mediana={statistics.median(diff_strategy_sim):.4f}\n")

    lines.append(
        f"Primeira frase (até a primeira vírgula/ponto/ponto-e-vírgula): "
        f"**{len(exact_counts)}** moldes distintos por correspondência textual exata "
        f"(entre {n_total} séries). Os 5 mais comuns por correspondência exata:\n"
    )
    lines.append("| primeira frase (exata) | n_series |")
    lines.append("|---|---|")
    for fc, c in exact_sorted[:5]:
        lines.append(f"| \"{fc}\" | {c} |")
    lines.append("")
    lines.append(
        f"Agrupando por quase-duplicata (Jaccard sobre palavras de conteúdo da primeira frase ≥ {CLUSTER_THRESHOLD}, "
        f"clusterização por componentes conexos): **{len(cluster_sizes)}** moldes distintos ao todo; "
        f"**{molds_for_80}** moldes cobrem ≥80% da amostra ({cum}/{n_total} séries nesses {molds_for_80} moldes). "
        f"Tamanhos dos moldes, do maior ao menor: {cluster_sizes}.\n"
    )

    lines.append("## Passo 4 — variação por dataset e por estratégia\n")
    lines.append("### 4.1 — similaridade Jaccard por dataset (mesma métrica do Passo 3.1, só dentro de cada dataset)\n")
    lines.append("| dataset | n_series | n_pares | média | mediana |")
    lines.append("|---|---|---|---|---|")
    for dataset in C.DATASETS:
        s = per_dataset_stats[dataset]
        mean_s = f"{s['mean']:.4f}" if s["mean"] is not None else "n/a (0 pares)"
        median_s = f"{s['median']:.4f}" if s["median"] is not None else "n/a"
        lines.append(f"| {dataset} | {s['n_series']} | {s['n_pairs']} | {mean_s} | {median_s} |")
    lines.append("")

    lines.append(
        "### 4.2 — similaridade da primeira frase por mecanismo de formação do pool da estratégia final\n"
    )
    lines.append(
        "Mecanismo = ferramenta que criou o `pool` referenciado por `best_strategy_params` daquela série "
        "(`prune_redundant`, `select_stable`, `select_top_k`), `pool_full` se a estratégia usa o pool cheio sem "
        "nenhuma dessas ferramentas, ou `outro_ou_nao_rastreado` se o pool não foi encontrado na trajetória "
        "registrada. Verificado manualmente: os pools em `outro_ou_nao_rastreado` são nomeados `pool1`/`pool2`/"
        "`pool3` (ex.: `ANP_MONTHLY` série 1, `pool1`; série 41, `pool3`) -- esses nomes batem com os pools "
        "pré-semeados automaticamente por `select_stable` antes do loop ReAct começar (`orchestrator_react/"
        "pool.py:SEED_STABLE_POOLS`), não com uma ação dentro da própria trajetória do agente, por isso não "
        "aparecem em `react_trajectory_json`.\n"
    )
    lines.append("| mecanismo | n_series | n_pares internos | média Jaccard (1ª frase) dentro do mecanismo |")
    lines.append("|---|---|---|---|")
    for mech, stats in sorted(within_mechanism_sim.items(), key=lambda kv: -kv[1]["n_series"]):
        mean_s = f"{stats['mean_first_clause_jaccard']:.4f}" if stats["mean_first_clause_jaccard"] is not None else "n/a"
        lines.append(f"| {mech} | {stats['n_series']} | {stats['n_pairs']} | {mean_s} |")
    lines.append("")
    lines.append(
        f"Média Jaccard da 1ª frase entre pares de mecanismos **diferentes** "
        f"(n={len(across_mechanism_sims)} pares): **{mean_across_mechanism:.4f}**.\n"
    )

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    print(f"salvo em {out_path}")
    print(f"n_total={n_total}, top ngram={top15[0] if top15 else None}")


if __name__ == "__main__":
    main()
