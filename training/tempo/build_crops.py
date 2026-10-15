"""Build the printed tempo training crops from the SymbTr Mus2 PDFs (run extract_marks first).

    python -m training.tempo.build_crops

For every piece whose printed marks all matched the .mu2 tempos, every page is drawn as the service draws it
(1920 px wide), its staffs found, the band above each staff cut into ink clusters (homr.tempo_reader), and the
clusters saved as crops: the one that holds a tempo mark gets the target 'U8 1 6 8', a sample of the others an
empty target. Every 10th piece goes to val, so no piece is in both.
Writes datasets/tempo-mus2/crops/*.png and labels_{train,val}.jsonl.
"""
import json
import os
import random

import cv2
import numpy as np
import pymupdf

from homr.tempo_reader import bands, clusters, crop

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SYMBTR = os.path.join(ROOT, "datasets", "SymbTr-2.0.0")
OUT = os.path.join(ROOT, "datasets", "tempo-mus2")
UNIT = {"1/1": "U1", "1/2": "U2", "1/4": "U4", "1/8": "U8", "1/16": "U16"}
NEG_PER_PAGE = 6  # most clusters are lyrics and headings; keep a sample of them


def find_staffs_printed(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Staffs of a clean printed page, as (x0, y0, x1, y1) of their five lines: rows that are mostly ink."""
    ink = gray < 128
    rows = np.where(ink.sum(axis=1) > 0.45 * gray.shape[1])[0]
    lines: list[list[int]] = []
    for r in rows:
        if lines and r - lines[-1][-1] <= 2:
            lines[-1].append(r)
        else:
            lines.append([r])
    centers = [int(np.mean(line)) for line in lines]
    staffs, i = [], 0
    while i + 4 < len(centers):
        five = centers[i:i + 5]
        gaps = np.diff(five)
        if gaps.min() > 4 and gaps.max() < 1.5 * gaps.min():
            cols = np.where(ink[five[2]])[0]
            staffs.append((int(cols.min()), five[0], int(cols.max()), five[4]))
            i += 5
        else:
            i += 1
    return staffs


def main() -> None:
    marks = json.load(open(os.path.join(OUT, "marks.json"), encoding="utf-8"))
    names = sorted(n for n, m in marks.items() if m["isaretler"] and all("birim" in x for x in m["isaretler"]))
    os.makedirs(os.path.join(OUT, "crops"), exist_ok=True)
    files = {s: open(os.path.join(OUT, f"labels_{s}.jsonl"), "w", encoding="utf-8") for s in ("train", "val")}
    rng = random.Random(0)
    found = missed = 0
    for index, name in enumerate(names):
        split = "val" if index % 10 == 0 else "train"
        by_page: dict[int, list[dict]] = {}
        for m in marks[name]["isaretler"]:
            by_page.setdefault(m["sayfa"], []).append(m)
        with pymupdf.open(os.path.join(SYMBTR, "pdf", name + ".pdf")) as doc:
            for page_no, page in enumerate(doc, 1):
                zoom = 1920 / page.rect.width
                pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), colorspace=pymupdf.csGRAY)
                gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
                staffs = find_staffs_printed(gray)
                if not staffs:
                    continue
                targets = [([v * zoom for v in m["kutu"]], f"{UNIT[m['birim']]} {' '.join(str(m['deger']))}")
                           for m in by_page.get(page_no, []) if m["birim"] in UNIT]
                hit: set[int] = set()
                negatives = []
                for band, staff in zip(bands(staffs, gray.shape[1]), staffs, strict=True):
                    height = staff[3] - staff[1]
                    for box in clusters(gray, band, height):
                        label = ""
                        for k, (t, text) in enumerate(targets):
                            cx, cy = (t[0] + t[2]) / 2, (t[1] + t[3]) / 2
                            if box[0] <= cx <= box[2] and box[1] <= cy <= box[3] and k not in hit:
                                label = text
                                hit.add(k)
                        if label:
                            path = f"crops/{name}__p{page_no}__{box[0]}_{box[1]}.png"
                            cv2.imwrite(os.path.join(OUT, path), crop(gray, box, height))
                            files[split].write(json.dumps({"resim": path, "hedef": label, "eser": name}) + "\n")
                        else:
                            negatives.append((box, height))
                for box, height in rng.sample(negatives, min(NEG_PER_PAGE, len(negatives))):
                    path = f"crops/{name}__p{page_no}__{box[0]}_{box[1]}.png"
                    cv2.imwrite(os.path.join(OUT, path), crop(gray, box, height))
                    files[split].write(json.dumps({"resim": path, "hedef": "", "eser": name}) + "\n")
                found += len(hit)
                missed += len(targets) - len(hit)
    for f in files.values():
        f.close()
    print("tempo clusters found", found, "missed", missed)


if __name__ == "__main__":
    main()
