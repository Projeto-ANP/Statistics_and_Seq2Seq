# FASE 0 — Avaliar System One (Jev-class) no SERVIDOR

Objetivo: escolher com NÚMERO o substituto do LAYA zero-shot nos papéis de
gate/verificador do ReAct. Tudo abaixo roda SÓ no servidor
(`lucas.castro@anp4-cisia`, conda env `agno` ativo).

## 1. Baixar e subir os candidatos

### 1a. kev (Apache-2.0, família Jev-like treinável)

```bash
pip install uv            # se `uv` não existir
git clone https://github.com/jaredpalmer/kev.git
cd kev
uv sync --extra serve      # cria venv com torch CUDA; primeira vez demora
nohup uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b --port 8009 \
  > ~/Statistics_and_Seq2Seq/logs/kev_serve.log 2>&1 &
cd ~/Statistics_and_Seq2Seq
# primeiro run baixa adapter + base Qwen3.5-4B (≈10GB) automaticamente
curl -s localhost:8009/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Shoes arrived late.", "model": "kev-latest",
  "questions": {"ok": {"type": "noul", "instructions": "Is this urgent?"}}}'
```

Opcional (velocidade): `kev-0.8b` na porta 8010 — `--run jaredpalmer/kev-0.8b --port 8010`.

### 1b. Eikos-4B (MIT, calibrado, Qwen3.5-4B)

```bash
huggingface-cli download caiovicentino1/Eikos-4B --local-dir ~/models/Eikos-4B
git clone https://github.com/caiovicentino/eikos.git
cd eikos
pip install -U vllm          # requer vLLM >= 0.30 (leitura de logits por letra)
bash serve_vllm.sh ~/models/Eikos-4B 8001
nohup python serve.py --model ~/models/Eikos-4B --vllm-url http://127.0.0.1:8001 --port 8000 \
  > ~/Statistics_and_Seq2Seq/logs/eikos_serve.log 2>&1 &
cd ~/Statistics_and_Seq2Seq
```

Se `vllm` falhar na instalação do servidor, alternar para o caminho PyTorch
do repo do eikos (`python local_demo.py` usa MPS — no servidor usar o
`serve.py` sem vllm se o repo suportar; senão me avise e eu escrevo um
adapter transformers direto com o readout de logits das letras).

### 1c. JevBench (benchmark citável, 534 decisões congeladas)

```bash
git clone https://github.com/fstandhartinger/jevbench.git
cd jevbench
pip install -e .
```

## 2. Rodar as avaliações

### 2a. NOSSO domínio (o que importa): perguntas do gate com rótulos reais

```bash
cd ~/Statistics_and_Seq2Seq
python3 JEV/eval_systemone.py --laya \
  --endpoints kev4b=http://127.0.0.1:8009 eikos4b=http://127.0.0.1:8000 \
  --limit 600 --out JEV/phase0_results.json
```

Mede sobre `JEV/data/gate_dataset.jsonl`: acurácia, Brier, ECE, confiança,
latência — de cada modelo nas MESMAS 3 perguntas por janela que o gate usa.
Roda ~10-20 min (LAY A + 2 servidores). Trazer `phase0_results.json` de volta.

### 2b. JevBench oficial (número citável no artigo)

```bash
cd ~/Statistics_and_Seq2Seq/jevbench
python -m jevbench.cli run \
  --tasks datasets/public/easy.jsonl,datasets/public/hard.jsonl,datasets/public/original.jsonl \
  --adapter typesafe --endpoint http://127.0.0.1:8009 --model kev-latest --key-env "" \
  --results ~/Statistics_and_Seq2Seq/JEV/phase0_jevbench_kev4b.jsonl \
  --raw-dir /tmp/raw_kev --cap-usd 0
python -m jevbench.cli run \
  --tasks datasets/public/easy.jsonl,datasets/public/hard.jsonl,datasets/public/original.jsonl \
  --adapter typesafe --endpoint http://127.0.0.1:8000 --model eikos4b --key-env "" \
  --results ~/Statistics_and_Seq2Seq/JEV/phase0_jevbench_eikos4b.jsonl \
  --raw-dir /tmp/raw_eikos --cap-usd 0
# referência atual: LAYA local
python -m jevbench.cli run \
  --tasks datasets/public/easy.jsonl,datasets/public/hard.jsonl,datasets/public/original.jsonl \
  --adapter laya_local --endpoint local --revision multilingual \
  --results ~/Statistics_and_Seq2Seq/JEV/phase0_jevbench_laya.jsonl \
  --raw-dir /tmp/raw_laya --cap-usd 0
# resumos
python -m jevbench.cli summarize --tasks datasets/public/easy.jsonl,datasets/public/hard.jsonl,datasets/public/original.jsonl --results ~/Statistics_and_Seq2Seq/JEV/phase0_jevbench_kev4b.jsonl
python -m jevbench.cli summarize --tasks datasets/public/easy.jsonl,datasets/public/hard.jsonl,datasets/public/original.jsonl --results ~/Statistics_and_Seq2Seq/JEV/phase0_jevbench_eikos4b.jsonl
python -m jevbench.cli summarize --tasks datasets/public/easy.jsonl,datasets/public/hard.jsonl,datasets/public/original.jsonl --results ~/Statistics_and_Seq2Seq/JEV/phase0_jevbench_laya.jsonl
```

Nota: o adapter `typesafe` cobra um preço por decisão (`--cap-usd 0` + sem
`--price-*` → o runner pede reserve explícito). Se reclamar de preço, rodar
com `--price-in-per-m 0 --price-out-per-m 0`.

## 3. O que trazer de volta (via git)

- `JEV/phase0_results.json`
- `JEV/phase0_jevbench_*.jsonl` + os resumos impressos no terminal
- trechos de `logs/kev_serve.log` / `logs/eikos_serve.log` se algo falhar

## 4. Decisão esperada

Escolher pelo nosso 2a (acurácia+ECE nas perguntas do gate) com o 2b como
número citável; depois plugar o vencedor no `LayaAgent.predict` (mesma API
de pergunta) nos braços gate/verificador e re-medir ETTH1.
