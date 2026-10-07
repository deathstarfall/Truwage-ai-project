#!/usr/bin/env python3
"""
Step 3 (optional, needs scores) - CoRe-style contrastive regression on VideoMAE features.
Idea from CoRe (github.com/yuxumin/CoRe): don't predict a score directly; compare a query
clip with an EXEMPLAR clip of the same task whose score is known, predict the score
DIFFERENCE, then  score(query) = score(exemplar) + predicted_difference.
(This is a simplified version: no group-aware attention, no multi-exemplar groups.)

1) py aqa_core_head.py --make-template      -> writes indic_skill_100/scores.csv
2) Fill the 'score' column (e.g. 1-10 by watching each clip). Leave blanks to skip clips.
3) py aqa_core_head.py                      -> trains + reports Spearman / MAE on held-out clips
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

BASE = Path(__file__).resolve().parent / "indic_skill_100"
FEAT, SCORES, MANIFEST = BASE / "features" / "features.npz", BASE / "scores.csv", BASE / "manifest.csv"


def task_map():
    m = {}
    if MANIFEST.exists():
        for r in csv.DictReader(open(MANIFEST, encoding="utf-8")):
            m[Path(r["file"]).stem] = r["task"]
    return m


def rank(x):
    return np.argsort(np.argsort(x)).astype(float)


def spearman(a, b):
    ra, rb = rank(a), rank(b)
    return float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 2 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-template", action="store_true")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--exemplars", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    d = np.load(FEAT)
    names = list(d["names"])

    if a.make_template:
        tm = task_map()
        with open(SCORES, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["clip", "task", "score"])
            for n in names:
                w.writerow([n, tm.get(n, "unknown"), ""])
        print(f"Wrote {SCORES}. Fill in the 'score' column, then run again without --make-template.")
        return

    rows = [r for r in csv.DictReader(open(SCORES, encoding="utf-8")) if r["score"].strip()]
    idx = {n: i for i, n in enumerate(names)}
    clips = [r["clip"] for r in rows if r["clip"] in idx]
    y = np.array([float(r["score"]) for r in rows if r["clip"] in idx])
    task = np.array([r["task"] for r in rows if r["clip"] in idx])
    X = d["video"][[idx[c] for c in clips]]
    rng = np.random.default_rng(a.seed); torch.manual_seed(a.seed)

    perm = rng.permutation(len(clips))
    n_te = max(3, len(clips) // 4)
    te, tr = perm[:n_te], perm[n_te:]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)

    # same-task exemplar lists (train clips only)
    pool = {t: [i for i in tr if task[i] == t] for t in set(task)}
    pairs_ok = [i for i in tr if len(pool[task[i]]) >= 2]
    if len(pairs_ok) < 8:
        raise SystemExit("Need >= ~8 scored training clips with another same-task clip. Score more clips.")

    d_in = Xt.shape[1]
    net = nn.Sequential(nn.Linear(3 * d_in, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    f = lambda q, e: net(torch.cat([q, e, q - e], -1)).squeeze(-1)  # predicted score difference

    for ep in range(1, a.epochs + 1):
        net.train()
        q_idx = np.array(pairs_ok)
        e_idx = np.array([rng.choice([j for j in pool[task[i]] if j != i]) for i in q_idx])
        loss = nn.functional.mse_loss(f(Xt[q_idx], Xt[e_idx]), yt[q_idx] - yt[e_idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 100 == 0:
            print(f"epoch {ep}  train MSE {loss.item():.3f}")

    net.eval(); preds, truth = [], []
    with torch.no_grad():
        for i in te:
            ex = pool.get(task[i], [])
            if not ex:
                continue
            ex = list(rng.choice(ex, size=min(a.exemplars, len(ex)), replace=False))
            p = np.mean([float(yt[j] + f(Xt[i:i + 1], Xt[j:j + 1])) for j in ex])
            preds.append(p); truth.append(y[i])
    if len(preds) < 3:
        raise SystemExit("Too few test clips with same-task exemplars. Score more clips.")
    preds, truth = np.array(preds), np.array(truth)
    print(f"\nTest clips: {len(preds)}   Spearman: {spearman(preds, truth):.3f}   MAE: {np.abs(preds - truth).mean():.3f}")
    print("NOTE: with a small mock set or made-up scores these numbers are only a pipeline check.")


if __name__ == "__main__":
    main()
