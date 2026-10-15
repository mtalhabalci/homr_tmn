"""Page test of the tempo reader on the held-out Mus2 pieces (every 10th, as in build_crops): every page drawn as
the service draws it, its staffs found, its tempo marks read; compared with the printed marks.

    python -m training.tempo.page_test
"""
import json
import os

import numpy as np
import pymupdf

from homr.tempo_reader import read_tempo_marks
from training.tempo.build_crops import OUT, SYMBTR, find_staffs_printed

UNIT = {"1/1": "whole", "1/2": "half", "1/4": "quarter", "1/8": "eighth", "1/16": "16th"}


def main() -> None:
    marks = json.load(open(os.path.join(OUT, "marks.json"), encoding="utf-8"))
    names = sorted(n for n, m in marks.items() if m["isaretler"] and all("birim" in x for x in m["isaretler"]))
    held_out = [n for i, n in enumerate(names) if i % 10 == 0]
    expected = right = wrong = missed = extra = 0
    for name in held_out:
        want = {(m["sayfa"], UNIT[m["birim"]], m["deger"]) for m in marks[name]["isaretler"]}
        got = set()
        with pymupdf.open(os.path.join(SYMBTR, "pdf", name + ".pdf")) as doc:
            for page_no, page in enumerate(doc, 1):
                zoom = 1920 / page.rect.width
                pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), colorspace=pymupdf.csGRAY)
                gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
                for m in read_tempo_marks(gray, find_staffs_printed(gray)):
                    got.add((page_no, m.unit + ("." if m.dotted else ""), m.per_minute))
        expected += len(want)
        right += len(want & got)
        for w in want - got:
            if any(g[0] == w[0] for g in got - want):
                wrong += 1
            else:
                missed += 1
        extra += len({g for g in got - want if not any(w[0] == g[0] for w in want - got)})
    print(f"pieces {len(held_out)} | marks {expected} | right {right} ({right / expected:.1%}) | wrong value {wrong} | "
          f"missed {missed} | invented {extra}")


if __name__ == "__main__":
    main()
