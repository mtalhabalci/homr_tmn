"""Train the tempo reader (homr.tempo_reader.TempoReader) on the Mus2 crops (build_crops), the refereed handwritten
crops (datasets/tempo-el, when there) and synthetic crops.

    python -m training.tempo.train [--epochs 30] [--seed 0]   keeps the best val weights in OMR_TEMPO_MODEL
                                                              (default training/architecture/tempo/tempo_okuyucu.pt)
    python -m training.tempo.train --test                     val scores of those weights

Score, per val set (Mus2, handwritten, and a fixed synthetic set that holds ranges): share of the tempo crops read
exactly (note value and number), and share of the non-mark crops read as a mark (false alarms). The best epoch is
the one with the largest mean difference over the sets, so that no kind of mark is given up for another.
"""
import json
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
from torch import nn

from homr.tempo_reader import TOKENS, WEIGHTS, TempoReader, decode, to_tensor, valid

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "datasets", "tempo-mus2")
HANDWRITTEN = os.path.join(ROOT, "datasets", "tempo-el")
INDEX = {t: i + 1 for i, t in enumerate(TOKENS)}  # 0 is the CTC blank


def load_rows(split: str, folder: str = DATA) -> list[tuple[str, str]]:
    rows = []
    path = os.path.join(folder, f"labels_{split}.jsonl")
    if not os.path.exists(path):
        return rows
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        rows.append((os.path.join(folder, r["resim"]), r["hedef"]))
    return rows


def val_sets() -> dict[str, list]:
    """Mus2 val, handwritten val (when there), and 600 synthetic crops drawn with a fixed seed (a third of the marks
    ranges)."""
    from training.tempo.synthetic import sample  # noqa: PLC0415

    rng = random.Random(999)
    sets = {"mus2": load_rows("val"), "synthetic": [sample(rng) for _ in range(600)]}
    if load_rows("val", HANDWRITTEN):
        sets["handwritten"] = load_rows("val", HANDWRITTEN)
    return sets


@torch.no_grad()
def evaluate(model: TempoReader, rows: list[tuple[str, str]], device: str) -> dict:
    model.eval()
    marks = right = alarms = negatives = 0
    for i in range(0, len(rows), 64):
        chunk = rows[i:i + 64]
        images = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) if isinstance(p, str) else p for p, _ in chunk]
        for (seq, _), (_, target) in zip(decode(model(to_tensor(images).to(device))), chunk, strict=True):
            read = " ".join(seq) if valid(seq) else ""
            if target:
                marks += 1
                right += read == target
            else:
                negatives += 1
                alarms += bool(read)
    return {"marks_right": round(right / max(1, marks), 4), "marks": marks,
            "false_alarms": round(alarms / max(1, negatives), 4), "non_marks": negatives}


def train(epochs: int, seed: int = 0) -> None:
    from training.tempo.synthetic import sample  # noqa: PLC0415  (fonts are needed only for training)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(seed)
    torch.manual_seed(seed)
    real = load_rows("train")
    positives = [r for r in real if r[1]]
    negatives = [r for r in real if not r[1]]
    hand = load_rows("train", HANDWRITTEN)
    hand_positives = [r for r in hand if r[1]]
    hand_negatives = [r for r in hand if not r[1]]
    # shares of a batch: Mus2 marks, Mus2 non-marks, handwritten marks, handwritten non-marks; the rest synthetic
    shares = (0.25, 0.15, 0.15, 0.10) if hand_positives else (0.35, 0.25, 0.0, 0.0)
    val = val_sets()
    model = TempoReader().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    steps_per_epoch, batch = 250, 32
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=epochs * steps_per_epoch)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    cache: dict[str, np.ndarray] = {}

    def real_image(path: str) -> np.ndarray:
        if path not in cache:
            cache[path] = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cache[path]
        if rng.random() < 0.3:
            img = cv2.GaussianBlur(img, (3, 3), rng.uniform(0.3, 0.9))
        return img

    best = -1.0
    os.makedirs(os.path.dirname(WEIGHTS), exist_ok=True)
    for epoch in range(epochs):
        model.train()
        start, total = time.time(), 0.0
        for _ in range(steps_per_epoch):
            items = []
            for _ in range(batch):
                u = rng.random()
                for pool, edge in zip((positives, negatives, hand_positives, hand_negatives),
                                      np.cumsum(shares), strict=True):
                    if u < edge:
                        path, target = rng.choice(pool)
                        items.append((real_image(path), target))
                        break
                else:
                    items.append(sample(rng))
            x = to_tensor([im for im, _ in items]).to(device)
            targets = [[INDEX[t] for t in target.split()] for _, target in items]
            logits = model(x)
            log_probs = logits.log_softmax(-1).permute(1, 0, 2)
            input_lengths = torch.full((len(items),), logits.shape[1], dtype=torch.long)
            target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
            flat = torch.tensor([k for t in targets for k in t], dtype=torch.long)
            loss = ctc(log_probs, flat, input_lengths, target_lengths)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            total += loss.item()
        scores = {name: evaluate(model, rows, device) for name, rows in val.items()}
        overall = float(np.mean([s["marks_right"] - s["false_alarms"] for s in scores.values()]))
        print(f"epoch {epoch + 1}: loss {total / steps_per_epoch:.3f}  score {overall:.4f}  val {scores}  "
              f"({time.time() - start:.0f} s)", flush=True)
        if overall > best:
            best = overall
            torch.save(model.state_dict(), WEIGHTS)


if __name__ == "__main__":
    if "--test" in sys.argv:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = TempoReader().to(device)
        model.load_state_dict(torch.load(WEIGHTS, map_location=device))
        print({name: evaluate(model, rows, device) for name, rows in val_sets().items()})
    else:
        train(int(sys.argv[sys.argv.index("--epochs") + 1]) if "--epochs" in sys.argv else 30,
              int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 0)
