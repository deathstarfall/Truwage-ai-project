#!/usr/bin/env python3
"""
Step 2 - Audio-video contrastive alignment (CLIP-style InfoNCE).
Learns small projection heads on top of FROZEN VideoMAE and wav2vec2 features so that
a clip's video embedding is closest to its OWN audio embedding.

Run:  py aqa_contrastive.py
Out:  indic_skill_100/features/contrastive_heads.pt
      indic_skill_100/features/alignment_scores.csv   (video-audio cosine per clip)
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FEAT = Path(__file__).resolve().parent / "indic_skill_100" / "features"


def head(d_in, d_out=256):
    return nn.Sequential(nn.Linear(d_in, 512), nn.GELU(), nn.Linear(512, d_out))


def info_nce(v, a, logit_scale):
    v, a = F.normalize(v, dim=-1), F.normalize(a, dim=-1)
    logits = logit_scale.exp() * v @ a.T
    y = torch.arange(len(v), device=v.device)
    return (F.cross_entropy(logits, y) + F.cross_entropy(logits.T, y)) / 2


@torch.no_grad()
def recall(v, a, ks=(1, 5)):
    sim = F.normalize(v, dim=-1) @ F.normalize(a, dim=-1).T
    rank = sim.argsort(dim=1, descending=True)
    truth = torch.arange(len(v)).unsqueeze(1)
    pos = (rank == truth).float().argmax(dim=1)
    return {f"R@{k}": float((pos < k).float().mean()) for k in ks if k <= len(v)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    d = np.load(FEAT / "features.npz")
    keep = d["has_audio"]
    names, V, A = d["names"][keep], d["video"][keep], d["audio"][keep]
    n = len(names)
    print(f"{n} clips with usable audio (of {len(d['names'])})")
    if n < 12:
        raise SystemExit("Need at least ~12 clips WITH audio. Download more clips that contain sound.")

    perm = rng.permutation(n)
    n_val = max(4, n // 5)
    val, tr = perm[:n_val], perm[n_val:]
    mu_v, sd_v = V[tr].mean(0), V[tr].std(0) + 1e-6   # standardise with TRAIN stats only
    mu_a, sd_a = A[tr].mean(0), A[tr].std(0) + 1e-6
    Vt = torch.tensor((V - mu_v) / sd_v, dtype=torch.float32)
    At = torch.tensor((A - mu_a) / sd_a, dtype=torch.float32)

    hv, ha = head(Vt.shape[1]), head(At.shape[1])
    scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))
    opt = torch.optim.AdamW(list(hv.parameters()) + list(ha.parameters()) + [scale], lr=a.lr, weight_decay=1e-2)

    for ep in range(1, a.epochs + 1):
        hv.train(); ha.train()
        idx = torch.tensor(rng.permutation(tr))
        for s in range(0, len(idx), a.batch):
            b = idx[s:s + a.batch]
            if len(b) < 2:
                continue
            loss = info_nce(hv(F.dropout(Vt[b], 0.1)), ha(F.dropout(At[b], 0.1)), scale)
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                scale.clamp_(0, np.log(100))
        if ep % 50 == 0 or ep == 1:
            hv.eval(); ha.eval()
            with torch.no_grad():
                vi = torch.tensor(val)
                r = recall(hv(Vt[vi]), ha(At[vi]))
            print(f"epoch {ep:4d}  loss {loss.item():.3f}  val {r}  (chance R@1 = {1/len(val):.2f})")

    hv.eval(); ha.eval()
    with torch.no_grad():
        sims = (F.normalize(hv(Vt), dim=-1) * F.normalize(ha(At), dim=-1)).sum(-1).numpy()
    with open(FEAT / "alignment_scores.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["clip", "split", "audio_video_cosine"])
        vs = set(val.tolist())
        for i, nm in enumerate(names):
            w.writerow([nm, "val" if i in vs else "train", f"{sims[i]:.4f}"])
    torch.save({"video_head": hv.state_dict(), "audio_head": ha.state_dict(),
                "stats": dict(mu_v=mu_v, sd_v=sd_v, mu_a=mu_a, sd_a=sd_a)}, FEAT / "contrastive_heads.pt")
    print("Saved contrastive_heads.pt and alignment_scores.csv")


if __name__ == "__main__":
    main()
