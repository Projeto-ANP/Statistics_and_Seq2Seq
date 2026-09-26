#!/usr/bin/env python3
"""KevClient — cliente HTTP do kev (família Jev-like) com a MESMA interface
`predict(state_text, question)` do LayaAgent, para ser drop-in no
`run_gate_pass_windows` e nos loops.

O kev responde no formato TypeSafe-compatível:

    POST {url}/v1/systemone
    {"state": <texto>, "model": "kev-latest", "questions": {qkey: {type, instructions, ...}}}

e devolve `{"answers": {qkey: {"type": "noul", "noul": 0.41}, ...}}` — exatamente
o shape que `run_gate_pass_windows` já lê (`out["answers"]["transfer"]["noul"]`).

Fase 0 mediu (1.791 decisões no nosso domínio): argmax do P do kev escolhe o
vencedor do teste melhor que o argmin (−0.0061) e MUITO melhor que o LAYA
zero-shot (+0.0083). Uso correto: RANKER (argmax), não probabilidade (ECE 0.43
fora do domínio de treino dele).
"""
from __future__ import annotations

import json
import urllib.request
from typing import Any, Dict


class KevClient:
    name = "kev"

    def __init__(self, url: str = "http://127.0.0.1:8009",
                 model: str = "kev-latest", timeout_s: float = 300.0) -> None:
        self.url = url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def predict(self, state_text: str, question: Dict[str, Any]) -> Dict[str, Any]:
        """Mesma assinatura do `LayaAgent.predict`: devolve
        `{"answers": {qkey: {...}}, ...}`. Uma passada responde todas as
        perguntas do dict (o kev processa o estado uma vez)."""
        body = {"state": state_text, "model": self.model, "questions": question}
        req = urllib.request.Request(
            f"{self.url}/v1/systemone",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def ping(self) -> Dict[str, Any]:
        return self.predict("ping", {
            "ok": {"type": "noul", "instructions": "Is this a ping?"},
        })


def main() -> int:
    import argparse
    p = argparse.ArgumentParser(description="Sanidade do KevClient")
    p.add_argument("--url", default="http://127.0.0.1:8009")
    args = p.parse_args()
    client = KevClient(args.url)
    out = client.ping()
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
