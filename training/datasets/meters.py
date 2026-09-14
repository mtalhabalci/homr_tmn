"""Time signatures the corpus has too few of, set onto real first staffs.

A time signature is printed once per work, on its first staff, so a usul the
corpus holds in two works is seen twice in training. The model then reads the
common ones and guesses the rest: 13/4, 14/8, 15/8 and 16/8 all came out as
10/8, and 16/8 and 15/8 are not in the training works at all.

The digits are only digits, and Mus2 sets them the same way whatever they
say. So a first staff is rendered with its own time signature taken off --
the digits' pixels replaced by the staff lines under them -- and another set in
its place in Mus2's own digits, centred where the old ones stood, the upper
number on the upper half of the staff and the lower on the lower half. The
label changes by that one symbol. The notes that follow no longer fill the
new measure exactly; the reader is being taught to read the digits, and the
picture and its label still agree.

Other engravers' digits are not Mus2's: Üsküdar Musiki Cemiyeti's pages, set
in a Finale-style font, had their 10/8 read as 16/8 in six works of nine. So
the new digits can also come from the SMuFL fonts, whose time-signature
digits every one of them carries.

    python -m training.datasets.meters --split train --per-class 100 --digit-fonts mixed
    python -m training.datasets.meters --split test --per-class 10
    python -m training.datasets.meters --split test --per-class 10 --suffix _fonts \
        --digit-fonts Bravura,FinaleMaestro,Gootville,Leland,MScore,MuseJazz,FinaleBroadway
"""

import argparse
import collections
import glob
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
from training.datasets.courtesy import (
    CLEARANCE,
    INK,
    MARGIN,
    _dilate,
    _mus2_fonts,
    _pixels,
    font_with,
    glyph_tint,
    staves_of,
    take_off,
    without_text,
)
from training.datasets.fonts import TRAIN_FONTS, font_buffer
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


# SMuFL's time-signature digits, 0 to 9.
TIME_SIG_ZERO = 0xE080


_parsed: dict[int, fitz.Font] = {}


def set_smufl_number(page: fitz.Page, text: str, centre: float, baseline: float, size: float, face: bytes) -> None:
    """A number in a SMuFL font's time-signature digits, which stand centred on their baseline.

    The page must already carry the font as "timesig": a SMuFL font is most of
    a megabyte, and embedding it for every number is what made this slow.
    """
    font = _parsed.get(id(face))
    if font is None:
        font = _parsed[id(face)] = fitz.Font(fontbuffer=face)
    codes = [TIME_SIG_ZERO + int(c) for c in text]
    widths = [font.glyph_advance(code) * size for code in codes]
    x = centre - sum(widths) / 2
    for code, width in zip(codes, widths):
        page.insert_text(fitz.Point(x, baseline), chr(code), fontname="timesig", fontsize=size, color=(0, 0, 0))
        x += width


def new_first_staff(path: str, lines: list[float], page_number: int, digits: list[dict], meter: str,
                    fonts: dict[str, bytes], face: bytes | None = None) -> np.ndarray | None:
    """The first staff with its time signature replaced, or None if it will not fit cleanly.

    The new digits are Mus2's, or with face a SMuFL font's, set as SMuFL sets
    them: four staff spaces to the em, the upper number centred on the upper
    half of the staff and the lower on the lower half.

    The old digits are taken off the rendered staff, not out of the pdf: see
    without_text for what removing characters from the pdf did to the notes.
    """
    middle = (lines[0] + lines[-1]) / 2
    upper = [d for d in digits if d["origin"][1] <= middle + 0.5]
    lower = [d for d in digits if d["origin"][1] > middle + 0.5]
    if not upper or not lower:
        return None
    with fitz.open(path) as document:
        page = document[page_number]
        clip = fitz.Rect(0, lines[0] - MARGIN, page.rect.width, lines[-1] + MARGIN)
        dpi = round(target_page_width / page.rect.width * 72)
        size = page.rect.width, page.rect.height
        own = _mus2_fonts(document)
        original = _pixels(page, clip, dpi).copy()
        bare = without_text(page, clip, dpi)
    tint = np.zeros(original.shape[:2], dtype=bool)
    for digit in digits:
        buffer = font_with(own, ord(digit["c"]))
        drawn = buffer and glyph_tint(original, size, clip, dpi, buffer, ord(digit["c"]), digit["origin"], digit["size"])
        if not drawn:
            return None
        tint |= drawn[1]
    base = take_off(original, bare, tint)
    # The new digits alone, laid over the emptied staff.
    blank = fitz.open()
    sheet = blank.new_page(width=size[0], height=size[1])
    centre = (min(d["bbox"].x0 for d in digits) + max(d["bbox"].x1 for d in digits)) / 2
    numerator, denominator = meter.split("/")
    if face is None:
        set_number(sheet, numerator, centre, upper[0]["origin"][1], upper[0]["size"], fonts)
        set_number(sheet, denominator, centre, lower[0]["origin"][1], lower[0]["size"], fonts)
    else:
        height = lines[-1] - lines[0]
        sheet.insert_font(fontname="timesig", fontbuffer=face)
        set_smufl_number(sheet, numerator, centre, lines[0] + height / 4, height, face)
        set_smufl_number(sheet, denominator, centre, lines[0] + 3 * height / 4, height, face)
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
    parser.add_argument(
        "--rare-per-class", type=int, default=None,
        help="Staffs for the signatures Turkish scores hardly use -- a half or a sixteenth "
             "as the beat, or one beat to the measure. Defaults to --per-class.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--digit-fonts", default="Mus2",
        help="Comma-separated fonts to set the new digits in, one chosen per staff: Mus2 and "
             "any of the SMuFL fonts in fonts.FONT_URLS. 'mixed' is Mus2 for a third of the "
             "staffs and the training fonts for the rest.",
    )
    parser.add_argument("--suffix", default="", help="Appended to the output folder and index name.")
    parser.add_argument("--part", default=None, help="K/N: only every Nth time signature from the Kth, to run N at once.")
    parser.add_argument("--merge", action="store_true", help="Join the index files the parts wrote.")
    options = parser.parse_args()
    stem = os.path.join(out_root, f"index_{options.split}{options.suffix}")
    if options.merge:
        parts = sorted(glob.glob(stem + ".part*.txt"))
        lines_out = [line for path in parts for line in open(path, encoding="utf-8") if line.strip()]
        with open(stem + ".txt", "w", encoding="utf-8") as handle:
            handle.writelines(lines_out)
        for path in parts:
            os.remove(path)
        eprint(f"{len(lines_out)} staffs from {len(parts)} parts -> {stem}.txt")
        return
    targets, part = TARGETS, ""
    if options.part:
        k, n = (int(v) for v in options.part.split("/"))
        targets, part = TARGETS[k::n], f".part{k}"
        options.seed += 10 * k

    index = {"train": index_train, "val": index_val, "test": index_test}[options.split]
    works = staves_of(index)
    fonts = digit_fonts(sorted(staves_of(index_train)))
    if len(fonts) < 10:
        eprint(f"Only found digits {sorted(fonts)}")
    rng = random.Random(options.seed)
    # Its own generator, so that setting digits in Mus2 alone draws exactly
    # the staffs it always has.
    font_rng = random.Random(options.seed + 1)
    if options.digit_fonts == "mixed":
        choices = ["Mus2"] * (len(TRAIN_FONTS) // 2) + list(TRAIN_FONTS)
    else:
        choices = options.digit_fonts.split(",")
    faces = {name: None if name == "Mus2" else font_buffer(name) for name in set(choices)}
    folder = os.path.join(out_root, options.split + options.suffix)
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

    def wanted(meter: str) -> int:
        numerator, denominator = (int(part) for part in meter.split("/"))
        rare = denominator in (2, 16) or numerator == 1
        if rare and options.rare_per_class is not None:
            return options.rare_per_class
        return options.per_class

    for meter in targets:
        for attempt in range(wanted(meter) * 4):
            if made[meter] >= wanted(meter):
                break
            work, path, page_number, lines, digits, tokens = rng.choice(firsts)
            rows = [l.split() for l in open(os.path.join(git_root, tokens), encoding="utf-8") if l.split()]
            label = f"timeSignature_{meter}"
            if not any(r[0].startswith("timeSignature") for r in rows) or any(r[0] == label for r in rows):
                continue
            face = font_rng.choice(choices) if len(choices) > 1 else choices[0]
            image = new_first_staff(path, lines, page_number, digits, meter, fonts, faces[face])
            if image is None:
                continue
            named = "" if face == "Mus2" else f"-f{face}"
            base = os.path.join(folder, f"{work}-00-m{meter.replace('/', '_')}-{attempt}{named}")
            with open(base + ".tokens", "w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(" ".join([label, *row[1:]] if row[0].startswith("timeSignature") else row) + "\n")
            pixmap = fitz.Pixmap(fitz.csRGB, image.shape[1], image.shape[0], np.ascontiguousarray(image).tobytes(), False)
            pixmap.save(base + ".png")
            made[meter] += 1
            rel = lambda p: os.path.relpath(p, git_root).replace(os.sep, "/")  # noqa: E731
            lines_out.append(f"{rel(base + '.png')},{rel(base + '.tokens')}\n")
    with open(stem + part + ".txt", "w", encoding="utf-8") as handle:
        handle.writelines(lines_out)
    short = [meter for meter in targets if made[meter] < wanted(meter)]
    eprint(f"{len(lines_out)} staffs across {len(targets)} time signatures; short of target: {short}")


if __name__ == "__main__":
    main()
