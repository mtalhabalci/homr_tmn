"""Codas, and segnos in other engravers' hands, set over real printed staffs.

After segno_labels.py the printed corpus teaches the segno 5,895 times -- but always Mus2's segno, and it
never teaches a coda at all: Mus2 prints no coda sign and no "Coda" word anywhere in the 2,200 pdfs. The
model cannot write a sign it has never been shown, and a handwritten ⊕ has nothing to be matched against.

So a staff is rendered from its pdf and a SMuFL coda or segno is set over it where these signs stand:
over the first measure, over a rule mid-staff, or over the closing rule at the end -- in proportion to where
Mus2 puts its own segnos (777, 105 and 752 of them). The sign is set at the size SMuFL prescribes, four
staff spaces to the em, its lowest ink three quarters of a space above the top line. If it would touch any
ink already there -- a high note, a slur, a volta bracket -- it is lifted half a space, twice at most, and
failing that the staff is left out. Its token goes where segno_labels.py puts a segno.

The staff picture is otherwise the real rendered page, and the label is its real label with one token more.
Petaluma is kept out of training, as everywhere else, to test a font never seen.

    python -m training.datasets.nav_marks --split train --coda 1500 --segno 800
    python -m training.datasets.nav_marks --split test --coda 150 --segno 100 --fonts Petaluma,Bravura,Leland
"""

import argparse
import collections
import json
import os
import random

import fitz
import numpy as np

from homr.simple_logging import eprint
from training.datasets.convert_symbtr import (
    _staff_lines,
    dataset_root,
    git_root,
    index_test,
    index_train,
    index_val,
    symbtr_pdf,
    target_page_width,
)
from training.datasets.courtesy import CLEARANCE, INK, MARGIN, _dilate, _pixels, sign_pixels, staves_of
from training.datasets.fonts import TRAIN_FONTS, font_buffer
from training.datasets.page_notes import bar_rules, read_staff, uses_usual_encoding
from training.datasets.segno_labels import segno_at

out_root = os.path.join(dataset_root, "SymbTr-navmarks")
GLYPH = {"coda": 0xE048, "segno": 0xE047}
# Where Mus2 sets its own segnos: 777 over the first measure, 105 mid-staff, 752 over the closing rule.
SPOTS = (("bas", 777), ("orta", 105), ("son", 752))
LIFT_TRIES = 3


def _ink_rows(pixels: np.ndarray) -> np.ndarray:
    return np.flatnonzero((pixels.min(axis=2) < INK).any(axis=1))


def set_sign(page: fitz.Page, lines: list[float], centre: float, face: bytes, code: int
             ) -> np.ndarray | None:
    """The staff with the sign set over it at centre, or None if it cannot stand there cleanly."""
    clip = fitz.Rect(0, lines[0] - MARGIN, page.rect.width, lines[-1] + MARGIN)
    dpi = round(target_page_width / page.rect.width * 72)
    size = page.rect.width, page.rect.height
    scale = dpi / 72
    height = lines[-1] - lines[0]
    space = height / 4
    original = _pixels(page, clip, dpi).copy()
    occupied = original.min(axis=2) < INK
    advance = fitz.Font(fontbuffer=face).glyph_advance(code) * height
    x = centre - advance / 2
    # Where the sign's ink falls relative to its baseline is the font's business, so it is measured once
    # with the baseline on the top line, and the baseline then moved to put the lowest ink where it belongs.
    trial = sign_pixels(size, clip, dpi, face, code, (x, lines[0]), height)
    rows = _ink_rows(trial)
    if rows.size == 0:
        return None
    lowest = clip.y0 + (rows[-1] + 1) / scale          # page units
    for lift in range(LIFT_TRIES):
        target = lines[0] - (0.75 + 0.5 * lift) * space
        baseline = lines[0] + (target - lowest)
        alone = sign_pixels(size, clip, dpi, face, code, (x, baseline), height)
        ink = alone.min(axis=2) < INK
        if not ink.any():
            return None
        rows = _ink_rows(alone)
        if rows[0] <= 1:
            return None                                  # it would stick out of the top of the crop
        if not (_dilate(ink, CLEARANCE) & occupied).any():
            return np.minimum(original, alone)
    return None


def main() -> None:  # noqa: PLR0915
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--coda", type=int, default=1500)
    parser.add_argument("--segno", type=int, default=800)
    parser.add_argument("--fonts", default=",".join(TRAIN_FONTS))
    parser.add_argument("--seed", type=int, default=0)
    options = parser.parse_args()

    index = {"train": index_train, "val": index_val, "test": index_test}[options.split]
    works = staves_of(index)
    marked = json.load(open(os.path.join(dataset_root, "SymbTr-segno", "senyo_yerleri.json"), encoding="utf-8"))
    rng = random.Random(options.seed)
    fonts = options.fonts.split(",")
    faces = {name: font_buffer(name) for name in fonts}
    folder = os.path.join(out_root, options.split)
    os.makedirs(folder, exist_ok=True)

    # Base staffs only -- a picture straight from the pdf -- and none that already carries a segno.
    candidates = []
    for work, staves in works.items():
        for number, image, tokens in staves:
            name = f"{work}-{number:02d}"
            if "SymbTr-work-v5/" not in image or name in marked:
                continue
            candidates.append((work, number, tokens))
    rng.shuffle(candidates)
    eprint(f"{len(candidates)} candidate staffs in {options.split}")

    wanted = {"coda": options.coda, "segno": options.segno}
    made: collections.Counter = collections.Counter()
    skipped: collections.Counter = collections.Counter()
    lines_out = []
    documents: dict[str, fitz.Document] = {}
    spots, weights = zip(*SPOTS)
    for work, number, tokens in candidates:
        if all(made[k] >= wanted[k] for k in wanted):
            break
        kind = "coda" if made["coda"] < wanted["coda"] and (made["segno"] >= wanted["segno"]
                                                            or rng.random() < 0.65) else "segno"
        rows = [line.split() for line in open(os.path.join(git_root, tokens), encoding="utf-8") if line.split()]
        if any(row[0].lstrip("?") in ("segno", "coda") for row in rows):
            skipped["zaten işaretli"] += 1
            continue
        path = os.path.join(symbtr_pdf, work + ".pdf")
        if work not in documents:
            if len(documents) > 40:
                for document in documents.values():
                    document.close()
                documents.clear()
            documents[work] = fitz.open(path)
        document = documents[work]
        if not uses_usual_encoding(document):
            skipped["kodlama farklı"] += 1
            continue
        staves = [(page.number, lines) for page in document for lines in _staff_lines(page)]
        if number >= len(staves):
            skipped["porte pdf'te yok"] += 1
            continue
        page = document[staves[number][0]]
        lines = staves[number][1]
        heads = read_staff(page, lines)
        notes = sum(1 for row in rows if row[0].lstrip("?").startswith("note"))
        if not heads or len(heads) != notes:
            skipped["nota sayısı tutmuyor"] += 1
            continue
        space = (lines[-1] - lines[0]) / 4
        spot = rng.choices(spots, weights)[0]
        rules = [x for x, shape in bar_rules(page, lines, heads) if shape != "dot"]
        if spot == "bas":
            centre = heads[0]["x"] - 0.4 * space
        elif spot == "son":
            if not rules or rules[-1] <= heads[-1]["x"]:
                skipped["son çizgi yok"] += 1
                continue
            centre = rules[-1] - 0.2 * space
        else:
            inner = [x for x in rules if heads[0]["x"] < x < heads[-1]["x"]]
            if not inner:
                skipped["ortada çizgi yok"] += 1
                continue
            centre = rng.choice(inner) + 1.2 * space
        k = sum(1 for head in heads if head["x"] < centre)
        if (spot == "bas" and k != 0) or (spot == "son" and k != len(heads)) or (
                spot == "orta" and not 0 < k < len(heads)):
            skipped["konum tutarsız"] += 1
            continue
        at = segno_at(rows, k, spot)
        if at is None:
            skipped["etikette yer yok"] += 1
            continue
        font = rng.choice(fonts)
        image = set_sign(page, lines, centre, faces[font], GLYPH[kind])
        if image is None:
            skipped["temiz sığmadı"] += 1
            continue
        rows = rows[:at] + [[kind, ".", ".", ".", "."]] + rows[at:]
        base = os.path.join(folder, f"{work}-{number:02d}-n{kind}-{spot}-f{font}")
        with open(base + ".tokens", "w", encoding="utf-8") as handle:
            handle.writelines(" ".join(row) + "\n" for row in rows)
        pixmap = fitz.Pixmap(fitz.csRGB, image.shape[1], image.shape[0], np.ascontiguousarray(image).tobytes(), False)
        pixmap.save(base + ".png")
        made[kind] += 1
        made[f"{kind}-{spot}"] += 1
        rel = lambda p: os.path.relpath(p, git_root).replace(os.sep, "/")  # noqa: E731
        lines_out.append(f"{rel(base + '.png')},{rel(base + '.tokens')}\n")
    for document in documents.values():
        document.close()
    with open(os.path.join(out_root, f"index_{options.split}.txt"), "w", encoding="utf-8") as handle:
        handle.writelines(lines_out)
    eprint(f"{len(lines_out)} staffs: {dict(made)}")
    eprint(f"left out: {dict(skipped)}")


if __name__ == "__main__":
    main()
