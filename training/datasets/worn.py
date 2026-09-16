"""Accidentals as an old print wears them.

On old book and typewriter scans v11 still misread about one note in twenty,
and of the misreads a musician confirmed, three in five were the kind of a
sign: a 1-comma sharp taken for the 4-comma one, a turned flat for a struck
one, a sign invented where there was none, or one not seen. What tells these
signs apart is small -- one stem or two, two bars or three -- and wear takes
exactly that away: ink spreads until the bars run together, thin strokes
break, a stencil smears, a scan at low resolution blocks the shape out.

So a training staff with signs is rendered as Mus2 set it, or with its signs in
another engraver's font (fonts.redraw), and every sign on it -- key signature
and notes alike -- is worn the same way, as one print wears: ink spread, thinned
and broken, smudged, or coarsened. A few specks of ink go into the blank space
before notes that carry no sign, where one would stand, so that dirt is not
read as a sign. The label does not change: every sign still means what it meant.

    python -m training.datasets.worn --split train --part 0/4   (... 3/4)
    python -m training.datasets.worn --split train --merge
    python -m training.datasets.worn --split test
"""

import argparse
import collections
import glob
import os
import random

import cv2
import fitz
import numpy as np

from homr.simple_logging import eprint
from training.datasets.convert_symbtr import _staff_lines, dataset_root, git_root, index_test, index_train, index_val
from training.datasets.convert_symbtr import symbtr_pdf, target_page_width
from training.datasets.courtesy import INK, MARGIN, _pixels, read_rows, staves_of
from training.datasets.fonts import TRAIN_FONTS, glyph_for, redraw, signs_of
from training.datasets.page_notes import read_staff, uses_usual_encoding

out_root = os.path.join(dataset_root, "SymbTr-worn")
WEARS = ("spread", "thin", "smudge", "coarse")
# Half the staffs keep Mus2's own signs, half take another font's first.
OTHER_FONT = 0.5


def wear(patch: np.ndarray, kind: str, rng: random.Random) -> np.ndarray:
    """One grey patch around a sign, the sign worn one way.

    Only the sign's own ink wears. Wearing the whole patch thickened, broke or
    greyed the staff lines inside it too, and left a box around every sign --
    a mark no print has, and one the model could have learned to look for.
    A staff line is what runs across the whole patch; it is kept as it was.
    """
    ink = patch < INK
    line_rows = ink.mean(axis=1) >= 0.6
    line = np.zeros_like(ink)
    line[line_rows] = ink[line_rows]
    sign = ink & ~line
    if not sign.any():
        return patch
    # The sign lifted off, anti-aliased edge and all; the lines stay.
    around = cv2.dilate(sign.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    clean = patch.copy()
    clean[around & ~line_rows[:, None]] = 255
    alone = np.full_like(patch, 255)
    alone[around] = patch[around]
    if kind == "spread":
        size = rng.choice((2, 3))
        grown = cv2.dilate(sign.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))) > 0
        worn = np.where(grown, rng.randint(20, 70), 255).astype(np.uint8)
    elif kind == "thin":
        thinner = cv2.erode(sign.astype(np.uint8), np.ones((2, 2), np.uint8)) > 0
        # Erosion can take a hairline away entirely; then the sign stays whole.
        kept = thinner if thinner.sum() > 0.35 * sign.sum() else sign
        worn = np.where(kept, alone, 255).astype(np.uint8)
        ys, xs = np.nonzero(kept)
        for _ in range(rng.randint(1, 3) if len(ys) else 0):
            k = rng.randrange(len(ys))
            worn[max(ys[k] - 1, 0):ys[k] + 2, max(xs[k] - 1, 0):xs[k] + 1] = 255
    elif kind == "smudge":
        blurred = cv2.GaussianBlur(alone, (0, 0), rng.uniform(1.0, 1.8))
        worn = np.where(blurred < rng.randint(110, 175), np.minimum(blurred, 60), 255).astype(np.uint8)
    else:  # coarse: a low-resolution scan of the sign, enlarged back
        factor = rng.uniform(0.35, 0.6)
        small = cv2.resize(alone, (max(int(alone.shape[1] * factor), 2), max(int(alone.shape[0] * factor), 2)),
                           interpolation=cv2.INTER_AREA)
        worn = cv2.resize(small, (alone.shape[1], alone.shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.minimum(clean, worn)


def worn_staff(path: str, page_number: int, lines: list[float], rng: random.Random) -> tuple[np.ndarray | None, str]:
    """The staff with every sign worn one way and a few specks before signless notes; or None and why."""
    with fitz.open(path) as document:
        page = document[page_number]
        signs = signs_of(page, lines)
        if not signs:
            return None, "no signs"
        heads = read_staff(page, lines)
        clip = fitz.Rect(0, lines[0] - MARGIN, page.rect.width, lines[-1] + MARGIN)
        dpi = round(target_page_width / page.rect.width * 72)
        faces = {name: fitz.Font(fontbuffer=open(os.path.join(dataset_root, "fonts", name + ".otf"), "rb").read())
                 for name in TRAIN_FONTS} if rng.random() < OTHER_FONT else {}
        image = None
        font = "Mus2"
        if faces:
            able = [n for n, face in faces.items() if all(glyph_for(face, s["lift"]) for s in signs)]
            if able:
                font = rng.choice(able)
        if font == "Mus2":
            image = _pixels(page, clip, dpi).copy()
    if font != "Mus2":
        image, why = redraw(path, page_number, lines, font)
        if image is None:
            return None, f"font: {why}"
    scale = dpi / 72
    step = (lines[-1] - lines[0]) / 8
    grey = image.min(axis=2).astype(np.uint8)
    kind = rng.choice(WEARS)
    for sign in signs:
        box = sign["bbox"]
        # Another font's sign is set a little wider and on the notehead's side: a wider patch holds it.
        pad_x, pad_y = box.width * (0.45 if font != "Mus2" else 0.2), box.height * 0.15
        x0 = max(int((box.x0 - pad_x - clip.x0) * scale), 0)
        x1 = min(int((box.x1 + pad_x * (0.3 if sign.get("head") else 1) - clip.x0) * scale) + 1, grey.shape[1])
        y0 = max(int((box.y0 - pad_y - clip.y0) * scale), 0)
        y1 = min(int((box.y1 + pad_y - clip.y0) * scale) + 1, grey.shape[0])
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        grey[y0:y1, x0:x1] = wear(grey[y0:y1, x0:x1], kind, rng)
    # Specks where a sign would stand, before notes that have none.
    bare = [h for h in heads if h["sign"] is None]
    for head in rng.sample(bare, min(len(bare), rng.randint(0, 3))):
        cx = int((head["x"] - rng.uniform(0.9, 1.6) * step * 2 - clip.x0) * scale)
        cy = int((head["y"] + rng.uniform(-1.0, 1.0) * step - clip.y0) * scale)
        radius = rng.randint(1, 2)
        window = grey[max(cy - 4, 0):cy + 5, max(cx - 4, 0):cx + 5]
        if window.size and window.min() > INK and 0 <= cx < grey.shape[1] and 0 <= cy < grey.shape[0]:
            cv2.circle(grey, (cx, cy), radius, rng.randint(30, 110), -1)
    return np.repeat(grey[..., None], 3, axis=2), f"{kind}-{font}"


def _write(image: np.ndarray, tokens: str, base: str) -> None:
    rows = read_rows(os.path.join(git_root, tokens))
    with open(base + ".tokens", "w", encoding="utf-8") as handle:
        handle.writelines(" ".join(row) + "\n" for row in rows)
    cv2.imwrite(base + ".png", image)


def merge_parts(split: str) -> None:
    stem = os.path.join(out_root, f"index_{split}")
    parts = sorted(glob.glob(stem + ".part*.txt"))
    lines_out = [line for path in parts for line in open(path, encoding="utf-8") if line.strip()]
    with open(stem + ".txt", "w", encoding="utf-8") as handle:
        handle.writelines(lines_out)
    for path in parts:
        os.remove(path)
    eprint(f"{len(lines_out)} worn staffs from {len(parts)} parts -> {stem}.txt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--limit", type=int, default=None, help="Only N works.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--part", default=None, help="K/N: only every Nth work from the Kth, to run N at once.")
    parser.add_argument("--merge", action="store_true", help="Join the index files the parts wrote.")
    options = parser.parse_args()
    if options.merge:
        merge_parts(options.split)
        return

    works = staves_of({"train": index_train, "val": index_val, "test": index_test}[options.split])
    names = sorted(works)[: options.limit] if options.limit else sorted(works)
    part = ""
    if options.part:
        k, n = (int(v) for v in options.part.split("/"))
        names, part = names[k::n], f".part{k}"
        options.seed += k
    folder = os.path.join(out_root, options.split)
    os.makedirs(folder, exist_ok=True)
    rng = random.Random(options.seed)
    made: list[str] = []
    kinds: collections.Counter = collections.Counter()
    refused: collections.Counter = collections.Counter()
    rel = lambda p: os.path.relpath(p, git_root).replace(os.sep, "/")  # noqa: E731
    for number_of_work, work in enumerate(names):
        path = os.path.join(symbtr_pdf, work + ".pdf")
        with fitz.open(path) as document:
            if not uses_usual_encoding(document):
                continue
            staves = [(page.number, lines) for page in document for lines in _staff_lines(page)]
        for staff_number, _, tokens in works[work]:
            if staff_number >= len(staves):
                continue
            page_number, lines = staves[staff_number]
            image, how = worn_staff(path, page_number, lines, rng)
            if image is None:
                refused[how] += 1
                continue
            base = os.path.join(folder, f"{work}-{staff_number:02d}-w{how}")
            _write(image, tokens, base)
            made.append(f"{rel(base + '.png')},{rel(base + '.tokens')}\n")
            kinds[how.split("-")[0]] += 1
        if (number_of_work + 1) % 100 == 0:
            eprint(f"  {number_of_work + 1}/{len(names)} works, {len(made)} staffs")
    index_path = os.path.join(out_root, f"index_{options.split}{part}.txt")
    with open(index_path, "w", encoding="utf-8") as handle:
        handle.writelines(made)
    eprint(f"{len(made)} worn staffs -> {index_path}")
    eprint("By wear: " + ", ".join(f"{k} {n}" for k, n in kinds.most_common()))
    eprint("Left out: " + ", ".join(f"{why} {n}" for why, n in refused.most_common()))


if __name__ == "__main__":
    main()
