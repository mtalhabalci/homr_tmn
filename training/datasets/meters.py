"""Time signatures the corpus has too few of, set onto real first staffs.

A time signature is printed once per work, on its first staff, so a usul the
corpus holds in two works is seen twice in training. The model then reads the
common ones and guesses the rest: 13/4, 14/8, 15/8 and 16/8 all came out as
10/8, and 16/8 and 15/8 are not in the training works at all.

The digits are only digits, and Mus2 sets them the same way whatever they
say. So a first staff is rendered with its own time signature taken out of
the pdf -- the characters removed, the staff lines left -- and another set in
its place in Mus2's own digits, centred where the old ones stood, the upper
number on the upper half of the staff and the lower on the lower half. The
label changes by that one symbol. The notes that follow no longer fill the
new measure exactly; the reader is being taught to read the digits, and the
picture and its label still agree.

    python -m training.datasets.meters --split train
    python -m training.datasets.meters --split test --per-class 10
"""

import argparse
import collections
import os
import random

import fitz
import numpy as np

from homr.simple_logging import eprint
from homr.transformer.vocabulary import _time_denominators, _time_numerators
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
from training.datasets.courtesy import CLEARANCE, INK, MARGIN, _dilate, _mus2_fonts, _pixels, staves_of  # noqa: F401
from training.datasets.page_notes import read_staff, uses_usual_encoding

TARGETS = [f"{n}/{d}" for n in _time_numerators for d in _time_denominators]
out_root = os.path.join(dataset_root, "SymbTr-meters")


def digit_fonts(works: list[str]) -> dict[str, bytes]:
    """A Mus2 copy holding each digit; a pdf embeds only the digits it prints."""
    found: dict[str, bytes] = {}
    for work in works:
        if len(found) == 10:
            break
        with fitz.open(os.path.join(symbtr_pdf, work + ".pdf")) as document:
            for buffer in _mus2_fonts(document):
                font = fitz.Font(fontbuffer=buffer)
                for digit in "0123456789":
                    if digit not in found and font.has_glyph(ord(digit)):
                        found[digit] = buffer
    return found


def printed_digits(page: fitz.Page, lines: list[float], first_note: float) -> list[dict]:
    """The time signature's characters, with their boxes."""
    top, bottom = lines[0], lines[-1]
    step = (bottom - top) / 8
    band = fitz.Rect(0, top - MARGIN, page.rect.width, bottom + MARGIN)
    digits = []
    for block in page.get_text("rawdict", clip=band)["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                if "Mus2" not in span.get("font", ""):
                    continue
                for char in span.get("chars", []):
                    x, y = char["origin"]
                    if char["c"].isdigit() and x < first_note and top - step < y < bottom + step:
                        digits.append({"c": char["c"], "bbox": fitz.Rect(char["bbox"]), "origin": (x, y),
                                       "size": span["size"]})
    return digits


def set_number(page: fitz.Page, text: str, centre: float, baseline: float, size: float,
               fonts: dict[str, bytes]) -> None:
    widths = [fitz.Font(fontbuffer=fonts[c]).text_length(c, fontsize=size) for c in text]
    x = centre - sum(widths) / 2
    for c, width in zip(text, widths):
        name = f"digit{c}"
        page.insert_font(fontname=name, fontbuffer=fonts[c])
        page.insert_text(fitz.Point(x, baseline), c, fontname=name, fontsize=size, color=(0, 0, 0))
        x += width


def new_first_staff(path: str, lines: list[float], page_number: int, digits: list[dict], meter: str,
                    fonts: dict[str, bytes]) -> np.ndarray | None:
    """The first staff with its time signature replaced, or None if it will not fit cleanly."""
    middle = (lines[0] + lines[-1]) / 2
    upper = [d for d in digits if d["origin"][1] <= middle + 0.5]
    lower = [d for d in digits if d["origin"][1] > middle + 0.5]
    if not upper or not lower:
        return None
    def characters(page: fitz.Page, clip: fitz.Rect) -> list[dict]:
        return [c for b in page.get_text("rawdict", clip=clip)["blocks"] for l in b.get("lines", [])
                for s in l["spans"] for c in s.get("chars", []) if c["c"].strip()]

    with fitz.open(path) as document:
        page = document[page_number]
        clip = fitz.Rect(0, lines[0] - MARGIN, page.rect.width, lines[-1] + MARGIN)
        dpi = round(target_page_width / page.rect.width * 72)
        before = characters(page, clip)
        for digit in digits:
            page.add_redact_annot(digit["bbox"])
        page.apply_redactions(
            images=fitz.PDF_REDACT_IMAGE_NONE,
            graphics=fitz.PDF_REDACT_LINE_ART_NONE,
            text=fitz.PDF_REDACT_TEXT_REMOVE,
        )
        # The removal must take the digits and nothing else: a signature sign
        # whose box reaches into a digit's would go with it.
        if len(characters(page, clip)) != len(before) - len(digits):
            return None
        base = _pixels(page, clip, dpi).copy()
        size = page.rect.width, page.rect.height
    # The new digits alone, laid over the emptied staff.
    blank = fitz.open()
    sheet = blank.new_page(width=size[0], height=size[1])
    centre = (min(d["bbox"].x0 for d in digits) + max(d["bbox"].x1 for d in digits)) / 2
    numerator, denominator = meter.split("/")
    set_number(sheet, numerator, centre, upper[0]["origin"][1], upper[0]["size"], fonts)
    set_number(sheet, denominator, centre, lower[0]["origin"][1], lower[0]["size"], fonts)
    alone = _pixels(sheet, clip, dpi)
    blank.close()
    ink = alone.min(axis=2) < INK
    occupied = base.min(axis=2) < INK
    scale = dpi / 72
    for y in lines:
        row = int(round((y - clip.y0) * scale))
        occupied[max(row - 2, 0) : row + 3, :] = False
    if (_dilate(ink, CLEARANCE) & occupied).any():
        return None
    return np.minimum(base, alone)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--per-class", type=int, default=40, help="Staffs drawn for each time signature.")
    parser.add_argument("--seed", type=int, default=0)
    options = parser.parse_args()

    index = {"train": index_train, "val": index_val, "test": index_test}[options.split]
    works = staves_of(index)
    fonts = digit_fonts(sorted(staves_of(index_train)))
    if len(fonts) < 10:
        eprint(f"Only found digits {sorted(fonts)}")
    rng = random.Random(options.seed)
    folder = os.path.join(out_root, options.split)
    os.makedirs(folder, exist_ok=True)

    firsts = []
    for work in sorted(works):
        staves = works[work]
        if not staves or staves[0][0] != 0:
            continue
        path = os.path.join(symbtr_pdf, work + ".pdf")
        with fitz.open(path) as document:
            if not uses_usual_encoding(document):
                continue
            all_staves = [(page.number, lines) for page in document for lines in _staff_lines(page)]
            page_number, lines = all_staves[0]
            page = document[page_number]
            heads = read_staff(page, lines)
            if not heads:
                continue
            digits = printed_digits(page, lines, heads[0]["x"])
            if digits:
                firsts.append((work, path, page_number, lines, digits, staves[0][2]))
    eprint(f"{len(firsts)} first staffs with a printed time signature")

    made: collections.Counter = collections.Counter()
    lines_out = []
    for meter in TARGETS:
        for attempt in range(options.per_class * 4):
            if made[meter] >= options.per_class:
                break
            work, path, page_number, lines, digits, tokens = rng.choice(firsts)
            rows = [l.split() for l in open(os.path.join(git_root, tokens), encoding="utf-8") if l.split()]
            label = f"timeSignature_{meter}"
            if not any(r[0].startswith("timeSignature") for r in rows) or any(r[0] == label for r in rows):
                continue
            image = new_first_staff(path, lines, page_number, digits, meter, fonts)
            if image is None:
                continue
            base = os.path.join(folder, f"{work}-00-m{meter.replace('/', '_')}-{attempt}")
            with open(base + ".tokens", "w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(" ".join([label, *row[1:]] if row[0].startswith("timeSignature") else row) + "\n")
            pixmap = fitz.Pixmap(fitz.csRGB, image.shape[1], image.shape[0], np.ascontiguousarray(image).tobytes(), False)
            pixmap.save(base + ".png")
            made[meter] += 1
            rel = lambda p: os.path.relpath(p, git_root).replace(os.sep, "/")  # noqa: E731
            lines_out.append(f"{rel(base + '.png')},{rel(base + '.tokens')}\n")
    with open(os.path.join(out_root, f"index_{options.split}.txt"), "w", encoding="utf-8") as handle:
        handle.writelines(lines_out)
    short = [meter for meter in TARGETS if made[meter] < options.per_class]
    eprint(f"{len(lines_out)} staffs across {len(TARGETS)} time signatures; short of target: {short}")


if __name__ == "__main__":
    main()
