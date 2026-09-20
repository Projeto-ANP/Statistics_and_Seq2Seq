"""Uso do catalogo de acoes nas 680 series de orchestrator_react_v5 (gpt-oss:20b).

Conta, por ferramenta do catalogo (registry.TOOLS), em quantas series foi chamada
(qualquer chamada em `tools_called`, ok ou nao) e quantas com ok=True. Nomes fora do
catalogo (ex.: 'unparsed', 'evaluate') nao entram. Reporta o denominador do catalogo
completo (24), do catalogo com drop_redundant_combine_actions (20) e o "efetivo"
(descontando o que ja era retido: weights_ols com 3 janelas; weights_pooled_meta_model
quando o dataset nao treinou meta-modelo).
"""
import collections, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
sys.path.insert(0, C.REPO if hasattr(C, "REPO") else os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from orchestrator_react.registry import TOOLS, REDUNDANT_COMBINE_TOOLS

any_call, ok_call, n = collections.Counter(), collections.Counter(), 0
per_ds_pooled = {}
for ds in C.DATASETS:
    df = C.load_orchestrator(sys.argv[1] if len(sys.argv) > 1 else C.ORCHESTRATOR_GPTOSS, ds)
    for r in df.itertuples(index=False):
        n += 1
        tc = json.loads(r.tools_called) if isinstance(r.tools_called, str) else r.tools_called
        calls = tc.get("tools_called") if isinstance(tc, dict) else tc
        seen, seen_ok = set(), set()
        for c in calls or []:
            if c["tool"] in TOOLS:
                seen.add(c["tool"])
                if c["ok"]:
                    seen_ok.add(c["tool"])
        any_call.update(seen); ok_call.update(seen_ok)

names = list(TOOLS)
reduced = [t for t in names if t not in REDUNDANT_COMBINE_TOOLS]
never = lambda pool, cnt: [t for t in pool if cnt[t] == 0]
print(f"series: {n}")
print(f"{'ferramenta':28s} series com chamada | series com chamada ok")
for t in names:
    print(f"{t:28s} {any_call[t]:5d} | {ok_call[t]:5d}")
for label, pool in (("catalogo completo", names), ("com drop_redundant_combine_actions", reduced)):
    na, no = never(pool, any_call), never(pool, ok_call)
    print(f"\n{label}: {len(pool)} acoes | nunca chamadas: {len(na)} de {len(pool)} | nunca chamadas com sucesso: {len(no)} de {len(pool)}")
    print("  nunca chamadas:", na)
