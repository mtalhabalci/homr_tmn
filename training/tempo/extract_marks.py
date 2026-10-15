"""Find the printed tempo marks of the SymbTr Mus2 PDFs and match them to the .mu2 tempo rows.

    python -m training.tempo.extract_marks

Mus2 prints a tempo mark as one character of its music font (the note value, '=' drawn into the glyph) followed by
a bold Times number, e.g. 'z' + ' 168 ' (in the header followed by '⇒ 1 Dk 36 Sn', the duration). The .mu2 file
holds every tempo as a row '52 <tab> . <tab> 1 <tab> 8 <tab> 168' (an eighth, 168 per minute). The k-th printed
mark is matched to the k-th .mu2 tempo when the numbers agree; the Mus2 glyphs turn out to be y = quarter,
z = eighth, '{' = sixteenth, '|' = half.

Writes datasets/tempo-mus2/marks.json: per piece the .mu2 tempos and the printed marks (page, box in PDF points,
glyph, value, note value when matched).
"""
import collections
import glob
import json
import os
import re

import pymupdf

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SYMBTR = os.path.join(ROOT, "datasets", "SymbTr-2.0.0")
OUT = os.path.join(ROOT, "datasets", "tempo-mus2")


def mu2_tempos(path: str) -> list[dict]:
    out = []
    for line in open(path, encoding="utf-8", errors="replace").read().splitlines():
        f = line.split("\t")
        if f and f[0] == "52" and len(f) > 4:
            try:
                out.append({"birim": f"{f[2]}/{f[3]}", "deger": int(f[4])})
            except ValueError:
                pass
    return out


def marks_on_page(page: pymupdf.Page) -> list[dict]:
    spans = [s for b in page.get_text("dict")["blocks"] for line in b.get("lines", []) for s in line["spans"]]
    found = []
    for i, s in enumerate(spans):
        if not re.fullmatch(r"\s*\d{2,4}\s*", s["text"]) or "Bold" not in s["font"]:
            continue
        x0, y0, x1, y1 = s["bbox"]
        # the note glyph: a one-character span of the music font just left of the number, overlapping in height
        glyphs = [g for g in spans if g["font"].startswith("Mus2") and len(g["text"].strip()) == 1
                  and abs(g["bbox"][2] - x0) < 4 and g["bbox"][1] < y1 and g["bbox"][3] > y0]
        if not glyphs:
            continue
        g = min(glyphs, key=lambda g: abs(g["bbox"][2] - x0))
        box = [min(g["bbox"][0], x0), min(g["bbox"][1], y0), x1, max(g["bbox"][3], y1)]
        tail = [t for t in spans[i + 1:i + 3]
                if abs(t["bbox"][0] - box[2]) < 6 and ("⇒" in t["text"] or "Dk" in t["text"])]
        for t in tail:
            box = [box[0], min(box[1], t["bbox"][1]), max(box[2], t["bbox"][2]), max(box[3], t["bbox"][3])]
        found.append({"isaret": g["text"].strip(), "deger": int(s["text"]), "kutu": [round(v, 1) for v in box]})
    return found


def main() -> None:
    result, stats = {}, collections.Counter()
    for pdf in sorted(glob.glob(os.path.join(SYMBTR, "pdf", "*.pdf"))):
        name = os.path.splitext(os.path.basename(pdf))[0]
        mu2 = os.path.join(SYMBTR, "mu2", name + ".mu2")
        if not os.path.exists(mu2):
            stats["no mu2"] += 1
            continue
        tempos = mu2_tempos(mu2)
        marks = []
        with pymupdf.open(pdf) as doc:
            for page_no, page in enumerate(doc, 1):
                for m in marks_on_page(page):
                    marks.append(m | {"sayfa": page_no})
        for k, m in enumerate(marks):
            if k < len(tempos) and tempos[k]["deger"] == m["deger"]:
                m["birim"] = tempos[k]["birim"]
                stats["matched"] += 1
        stats["pieces"] += 1
        stats["printed"] += len(marks)
        result[name] = {"mu2": tempos, "isaretler": marks}
    os.makedirs(OUT, exist_ok=True)
    json.dump(result, open(os.path.join(OUT, "marks.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print(dict(stats))


if __name__ == "__main__":
    main()
