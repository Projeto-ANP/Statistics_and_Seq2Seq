#!/usr/bin/env python3
"""Fine-tune do LAYA como GATE de transferência (roda na GPU do servidor).

Adaptação single-GPU do notebook oficial
(laya_finetune_typed_decisions_2xT4_kaggle.ipynb): mesmo loop RLCD (GRPO com
`proper_reward` + cross-entropy suave), mesmos hiperparâmetros, sem DDP.

Rótulo: pergunta `noul` — "vai bater o piso das sementes no teste?" — com alvo
suavizado (0.9/0.1) a partir de `label` do gate_dataset.jsonl.

Uso (servidor, env agno + GPU):
  python JEV/finetune_laya_gate.py --holdout ETTM2 \
      --data JEV/data/gate_dataset.jsonl --output JEV/models/laya_gate_ettm2
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import torch  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

TARGET_INSTRUCTIONS = {
    "label": (
        "Will this candidate strategy beat every reference combination (FFORMA, ADE, "
        "and the seeded baselines) on the blind test window? Answer yes only if you "
        "expect it to be strictly better than all of them."
    ),
    "label_dyn": (
        "Is this candidate strategy the best available strategy for this series — "
        "better than every individual model, every combination (mean, median, dba, "
        "trimmed, weighted) and every seeded baseline? Answer yes only if you expect "
        "it to be strictly the best."
    ),
    "label_val": (
        "Is this candidate strategy the best available strategy for this series on "
        "the validation windows (nested leave-one-out)? Answer yes only if you "
        "expect it to be strictly the best on validation."
    ),
    "label_val_seed": (
        "Will this candidate strategy beat every seeded baseline on the validation "
        "windows (nested leave-one-out)? Answer yes only if you expect it to be "
        "strictly better than all of them on validation."
    ),
}

def make_q_spec(target: str) -> dict:
    return {"t": "noul", "ins": TARGET_INSTRUCTIONS[target], "crit": {}}

EPOCHS = 4
MICRO_BATCH = 8
GRAD_ACCUM = 4
GROUP_SIZE = 4
LR_ENCODER = 2.5e-5
LR_HEAD = 1.0e-4
SIGMA_START = 0.4
SIGMA_END = 0.1
CALIB_MAX = 400


def collate_train_batch(items, pad_id):
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, : len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)
    return {
        "input_ids": ids, "attention_mask": att, "marker_pos": mpos,
        "marker_mask": mmask, "target": target,
        "qtype": torch.tensor([it["qtype"] for it in items]),
        "label": torch.tensor([it["label"] for it in items]),
    }


def fit_one_temp(sel):
    if len(sel) < 10:
        return 1.0
    kmax = max(len(z) for z, _ in sel)
    Z = torch.full((len(sel), kmax), -1e4)
    T = torch.zeros((len(sel), kmax))
    for i, (z, t) in enumerate(sel):
        Z[i, : len(z)] = torch.tensor(z)
        T[i, : len(t)] = torch.tensor(t, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.clamp(log_t.exp(), 0.1, 10.0).item())


def build_items(data_path: str, holdout: str, tok, cfg, target: str):
    """Cada exemplo vira um item de treino: ids + markers + target suavizado."""
    from laya.common import build_sequence, render_options, QTYPES

    q_spec = make_q_spec(target)
    items = []
    skipped = 0
    with open(data_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row["dataset"] == holdout:
                continue
            p_true = 0.9 if int(row[target]) == 1 else 0.1
            target_vec = [1.0 - p_true, p_true]
            label = 1 if p_true > 0.5 else 0
            k = len(render_options(q_spec))
            seq, markers = build_sequence(
                tok, row["state"], q_spec, cfg["max_len"], cfg["head_max_len"]
            )
            if len(markers) != k or not seq:
                skipped += 1
                continue
            items.append({
                "ids": seq, "markers": markers,
                "qtype": QTYPES["noul"], "target": target_vec, "label": label,
            })
    print(f"itens de treino: {len(items)} (pulados: {skipped})")
    return items


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="JEV/data/gate_dataset.jsonl")
    ap.add_argument("--holdout", required=True, help="dataset deixado de fora (LOO)")
    ap.add_argument("--target", choices=["label", "label_dyn", "label_val",
                                          "label_val_seed"],
                    default="label_val",
                    help="label_val/label_val_seed = alvos SÓ de validação (treino); "
                         "label/label_dyn = alvos de teste (só análise)")
    ap.add_argument("--base-model", default="convaiinnovations/laya")
    ap.add_argument("--output", default=None)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.output is None:
        tag = {"label": "ref", "label_dyn": "dyn",
               "label_val": "val", "label_val_seed": "valseed"}[args.target]
        args.output = f"JEV/models/laya_gate_{args.holdout.lower()}_{tag}"

    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer
    from laya.agent import _fix_tokenizer_config
    from laya.common import build_model, proper_reward

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print(f"baixando modelo base {args.base_model} ...")
    model_dir = snapshot_download(args.base_model)
    _fix_tokenizer_config(model_dir)
    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))

    with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    cfg["gradient_checkpointing"] = True
    cfg["max_tokens_per_batch"] = 4096
    cfg["max_len"] = args.max_len
    cfg["head_max_len"] = 256

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
    weights = load_file(os.path.join(model_dir, "model.safetensors"))
    model.load_state_dict(weights, strict=True)
    model.encoder.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.head_checkpointing = True
    model.to(device)
    model.train()

    all_items = build_items(args.data, args.holdout, tok, cfg, args.target)
    if len(all_items) < 50:
        print("poucos itens de treino (<50) — verifique o --holdout e o dataset")
        return 2
    order = list(range(len(all_items)))
    random.Random(20260922).shuffle(order)
    n_calib = min(CALIB_MAX, len(all_items) // 10)
    calib_items = [all_items[i] for i in sorted(order[:n_calib])]
    train_items = [all_items[i] for i in sorted(order[n_calib:])]
    print(f"calibração: {len(calib_items)} | treino: {len(train_items)}")

    enc_params = [p for n, p in model.named_parameters() if "encoder." in n]
    head_params = [p for n, p in model.named_parameters() if "encoder." not in n]
    optimizer = torch.optim.AdamW(
        [{"params": enc_params, "lr": LR_ENCODER},
         {"params": head_params, "lr": LR_HEAD}],
        weight_decay=0.01,
    )
    total_updates = (len(train_items) // (MICRO_BATCH * GRAD_ACCUM)) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_updates), eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    t0 = time.time()
    for epoch in range(args.epochs):
        random.seed(42 + epoch)
        random.shuffle(train_items)
        epoch_loss, n_batches = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        accum = 0
        sigma = SIGMA_START + (SIGMA_END - SIGMA_START) * (epoch / max(1, args.epochs - 1))
        for b in range(0, len(train_items), MICRO_BATCH):
            chunk = train_items[b:b + MICRO_BATCH]
            batch = collate_train_batch(chunk, tok.pad_token_id)
            with torch.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                logits, act = model(
                    batch["input_ids"].to(device), batch["attention_mask"].to(device),
                    batch["marker_pos"].to(device), batch["marker_mask"].to(device),
                    batch["qtype"].to(device),
                )
            logits = logits.float()
            mask = batch["marker_mask"].to(device)
            k = mask.sum(-1, keepdim=True).float()
            target = batch["target"].to(device)
            eps = torch.randn((GROUP_SIZE,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), batch["qtype"].to(device), mask,
                                  w_sph=0.75, w_rps=1.0)
                adv = r - r.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)
            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
            loss_rl = -(adv * logp).mean()
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            loss = (loss_rl + loss_ce) / GRAD_ACCUM
            scaler.scale(loss).backward()
            accum += 1
            if accum % GRAD_ACCUM == 0 or (b + MICRO_BATCH) >= len(train_items):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            epoch_loss += loss.item() * GRAD_ACCUM
            n_batches += 1
            if n_batches % 50 == 0:
                print(f"  Epoch {epoch+1}/{args.epochs} | step {n_batches} | "
                      f"loss {loss.item()*GRAD_ACCUM:.4f} | reward {r.mean().item():.3f}")
        print(f"=== Epoch {epoch+1}/{args.epochs} em {time.time()-t0:.0f}s | "
              f"loss média {epoch_loss/max(1,n_batches):.4f} ===")
        ckpt_dir = os.path.join(args.output, "checkpoint_latest")
        os.makedirs(ckpt_dir, exist_ok=True)
        save_file({k: v.half().contiguous().cpu() for k, v in model.state_dict().items()},
                  os.path.join(ckpt_dir, "model.safetensors"))
        model.encoder.config.save_pretrained(os.path.join(ckpt_dir, "encoder"))
        tok.save_pretrained(os.path.join(ckpt_dir, "tokenizer"))
        with open(os.path.join(ckpt_dir, "checkpoint_meta.json"), "w") as f:
            json.dump({"epoch": epoch + 1, "total_epochs": args.epochs}, f, indent=2)

    # ── temperaturas de calibração (itens NUNCA usados no treino) ─────────────
    print("ajustando temperaturas de calibração ...")
    del optimizer, scheduler, scaler
    torch.cuda.empty_cache()
    model.eval()
    calib_preds = []
    with torch.no_grad():
        for c in range(0, len(calib_items), 16):
            cb = collate_train_batch(calib_items[c:c + 16], tok.pad_token_id)
            with torch.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                l_sub, _ = model(cb["input_ids"].to(device), cb["attention_mask"].to(device),
                                 cb["marker_pos"].to(device), cb["marker_mask"].to(device),
                                 cb["qtype"].to(device))
            l_np = l_sub.float().cpu().numpy()
            for rr, it in enumerate(calib_items[c:c + 16]):
                kk = len(it["markers"])
                calib_preds.append((it["qtype"], l_np[rr, :kk], it["target"]))
    fitted = [1.2, 1.2, 1.2]
    try:
        for qt in range(3):
            sel = [(z, t) for qty, z, t in calib_preds if qty == qt]
            if sel:
                fitted[qt] = fit_one_temp(sel)
        print("temperaturas (choice, score, noul):", [round(t, 3) for t in fitted])
    except Exception as exc:
        print("fallback de calibração:", exc)

    os.makedirs(args.output, exist_ok=True)
    save_file({k: v.half().contiguous().cpu() for k, v in model.state_dict().items()},
              os.path.join(args.output, "model.safetensors"))
    model.encoder.config.save_pretrained(os.path.join(args.output, "encoder"))
    tok.save_pretrained(os.path.join(args.output, "tokenizer"))
    cfg["fine_tuned"] = True
    cfg["model_name"] = f"laya-gate-{args.holdout.lower()}"
    cfg["temperature"] = fitted
    cfg.pop("temperature_by_options", None)
    with open(os.path.join(args.output, "rl_agent_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"checkpoint salvo em {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
