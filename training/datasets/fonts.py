"""The accidentals of real staffs, redrawn in other engravers' fonts.

Every sign the model has learned was drawn by Mus2. Other programs draw the
same AEU signs in their own hand: a page set in a Finale-style font gives the
1-comma sharp a hairline stem and slanted bars, and the model took it for the
ordinary sharp 11 times in 16 (the neyzen.com şehnaz longa). What a sign means
is carried by its shape -- one stem or two, two bars or three, a flat turned
round or struck through -- and the SMuFL fonts draw exactly these shapes
(U+E440-E447), so each can stand in for Mus2's own.

So a staff is rendered, its signs are taken off the picture -- each drawn alone
from the pdf's own Mus2 to find its pixels, which then take what lies under
them: staff and ledger lines, a slur -- and the same signs are set in their
place from another font, key signature and notes alike, one font for the
whole staff. The size is the one SMuFL prescribes, four staff spaces to the
em, and each sign stands on the staff degree Mus2 put it on. A sign among the
notes keeps its distance from its notehead: its right edge goes where Mus2's
was. A key signature sign keeps its centre. Where a new sign would touch
anything Mus2's did not, a smaller one is tried, and failing that the staff is
left out. The label does not change: every sign still means what it meant.

Petaluma is kept out of training, to measure a font the model has never seen.

    python -m training.datasets.fonts --split train --part 0/4   (... 3/4)
    python -m training.datasets.fonts --split train --merge
    python -m training.datasets.fonts --split train --more-rare
    python -m training.datasets.fonts --split test
"""

import argparse
import collections
import glob
import os
import random
import urllib.request

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
from training.datasets.courtesy import (
    CLEARANCE,
    INK,
    MARGIN,
    _dilate,
    _mus2_fonts,
    _pixels,
    font_with,
    glyph_tint,
    read_rows,
    sign_pixels,
    staves_of,
    take_off,
    without_text,
)
from training.datasets.page_notes import LEDGER_REACH, is_clef, lift_of_glyph, read_staff, uses_usual_encoding

font_dir = os.path.join(dataset_root, "fonts")
out_root = os.path.join(dataset_root, "SymbTr-fonts")

_MUSESCORE = "https://raw.githubusercontent.com/musescore/MuseScore/main/fonts/"
_STEINBERG = "https://raw.githubusercontent.com/steinbergmedia/"
# All under the SIL Open Font License, except the two Finale fonts, which
# MakeMusic licenses for free use and MuseScore ships.
FONT_URLS = {
    "Bravura": _STEINBERG + "bravura/master/redist/otf/Bravura.otf",
    "Petaluma": _STEINBERG + "petaluma/master/redist/otf/Petaluma.otf",
    "Leland": _MUSESCORE + "leland/Leland.otf",
    "Gootville": _MUSESCORE + "gootville/Gootville.otf",
    "MuseJazz": _MUSESCORE + "musejazz/MuseJazz.otf",
    "FinaleMaestro": _MUSESCORE + "finalemaestro/FinaleMaestro.otf",
    "FinaleBroadway": _MUSESCORE + "finalebroadway/FinaleBroadway.otf",
    "MScore": _MUSESCORE + "mscore/MScore.otf",
}
TRAIN_FONTS = ("Bravura", "FinaleMaestro", "Gootville", "Leland", "MScore", "MuseJazz", "FinaleBroadway")
UNSEEN_FONTS = ("Petaluma",)

# The AEU signs by what they mean.
SMUFL = {
    "flat8": 0xE440, "flat5": 0xE441, "flat4": 0xE442, "flat1": 0xE443,
    "sharp1": 0xE444, "sharp4": 0xE445, "sharp5": 0xE446, "sharp8": 0xE447,
    "N": 0xE261,
}
# The bakiye sharp is the ordinary sharp and the küçük mücenneb flat the
# ordinary flat; a font that has no AEU range still draws those two.
PLAIN = {"sharp4": 0xE262, "flat5": 0xE260}
# The sizes a sign is tried at, as a share of the size SMuFL prescribes.
SHRINK = (1.0, 0.9, 0.8)
# Signs too few among the notes to learn in every font from one setting each.
RARE = ("sharp1", "sharp5", "sharp8", "flat8")


def font_path(name: str) -> str:
    path = os.path.join(font_dir, name + ".otf")
    if not os.path.exists(path):
        os.makedirs(font_dir, exist_ok=True)
        eprint(f"Downloading {name}")
        urllib.request.urlretrieve(FONT_URLS[name], path)
    return path


def glyph_for(font: fitz.Font, lift: str) -> int | None:
    for table in (SMUFL, PLAIN):
        code = table.get(lift)
        if code is not None and font.has_glyph(code):
            return code
    return None


def signs_of(page: fitz.Page, lines: list[float]) -> list[dict]:
    """Every accidental character the staff carries, key signature and notes alike."""
    top, bottom = lines[0], lines[-1]
    step = (bottom - top) / 8
    band = fitz.Rect(0, top - MARGIN, page.rect.width, bottom + MARGIN)
    found = []
    for block in page.get_text("rawdict", clip=band)["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                if "Mus2" not in span.get("font", ""):
                    continue
                for char in span.get("chars", []):
                    code = ord(char["c"])
                    x, y = char["origin"]
                    glyph = {"code": code, "size": span["size"]}
                    lift = lift_of_glyph(code)
                    if lift is None or is_clef(glyph, lines):
                        continue
                    if not top - LEDGER_REACH * step <= y <= bottom + LEDGER_REACH * step:
                        continue
                    found.append({"code": code, "lift": lift, "bbox": fitz.Rect(char["bbox"]), "origin": (x, y),
                                  "x": char["bbox"][0], "size": span["size"]})
    return sorted(found, key=lambda s: s["x"])


def _same(a: dict, b: dict) -> bool:
    return a["code"] == b["code"] and abs(a["x"] - b["x"]) < 0.5 and abs(a["origin"][1] - b["origin"][1]) < 0.5


_buffers: dict[str, bytes] = {}


def font_buffer(name: str) -> bytes:
    if name not in _buffers:
        with open(font_path(name), "rb") as handle:
            _buffers[name] = handle.read()
    return _buffers[name]


def _shift(grey: np.ndarray, dx: int) -> np.ndarray:
    moved = np.full_like(grey, 255)
    if dx >= 0:
        moved[:, dx:] = grey[:, : grey.shape[1] - dx]
    else:
        moved[:, :dx] = grey[:, -dx:]
    return moved

def redraw(path: str, page_number: int, lines: list[float], font: str) -> tuple[np.ndarray | None, str]:
    """The staff with every accidental in the given font, or None and why it would not go cleanly.

    The old signs are taken off the rendered staff, not out of the pdf; see
    without_text for why.
    """
    new_font = font_buffer(font)
    face = fitz.Font(fontbuffer=new_font)
    with fitz.open(path) as document:
        page = document[page_number]
        found = signs_of(page, lines)
        if not found:
            return None, "no signs"
        codes = {s["lift"]: glyph_for(face, s["lift"]) for s in found}
        if None in codes.values():
            return None, "font lacks a sign"
        own = _mus2_fonts(document)
        heads = read_staff(page, lines)
        clip = fitz.Rect(0, lines[0] - MARGIN, page.rect.width, lines[-1] + MARGIN)
        dpi = round(target_page_width / page.rect.width * 72)
        size = page.rect.width, page.rect.height
        original = _pixels(page, clip, dpi).copy()
        bare = without_text(page, clip, dpi)
    scale = dpi / 72
    # Mus2 sometimes sets a signature sign twice on the same spot; it shows once.
    signs: list[dict] = []
    for sign in found:
        if not any(_same(sign, other) for other in signs):
            signs.append(sign)
    tint = np.zeros(original.shape[:2], dtype=bool)
    for sign in signs:
        sign["head"] = next((h for h in heads if h["sign"] and _same(sign, h["sign"])), None)
        buffer = font_with(own, sign["code"])
        drawn = buffer and glyph_tint(original, size, clip, dpi, buffer, sign["code"], sign["origin"], sign["size"])
        if not drawn:
            return None, "old sign not drawn where the page has it"
        sign["ink"], sign["tint"] = drawn
        tint |= sign["tint"]
    image = take_off(original, bare, tint)

    # Staff lines and ledger lines run under or up to a sign in any font.
    occupied = image.min(axis=2) < INK
    staff_height = lines[-1] - lines[0]
    space = staff_height / 4
    ledgers = [lines[0] - k * space for k in range(1, 5)] + [lines[-1] + k * space for k in range(1, 5)]
    for y in list(lines) + ledgers:
        row = int(round((y - clip.y0) * scale))
        occupied[max(row - 2, 0) : max(row + 3, 0), :] = False
    for sign in signs:
        old_columns = np.nonzero(sign["ink"].any(axis=0))[0]
        old_left, old_right = old_columns[0], old_columns[-1]
        around = occupied.copy()
        if sign["head"] is not None:
            # Only what lies left of its own notehead can be in the sign's way.
            around[:, max(int((sign["head"]["x"] - clip.x0) * scale) - 1, 0) :] = False
        # Nor what Mus2's own sign already stood on, such as a slur.
        around &= ~_dilate(sign["tint"], CLEARANCE + 1)
        # Most of these fonts draw a wider sign than Mus2, which spaced the
        # notes for its own; where it will not go, a smaller one is tried, as
        # engravers set accidentals smaller in a crowded passage.
        for shrink in SHRINK:
            origin = (sign["bbox"].x0, sign["origin"][1])
            new = sign_pixels(size, clip, dpi, new_font, codes[sign["lift"]], origin, staff_height * shrink).min(axis=2)
            new_columns = np.nonzero((new < INK).any(axis=0))[0]
            if not len(new_columns):
                return None, "new sign drew nothing"
            if sign["head"] is not None:
                dx = old_right - new_columns[-1]
            else:
                dx = round((old_left + old_right - new_columns[0] - new_columns[-1]) / 2)
            new = _shift(new, int(dx))
            ink = new < INK
            if not (_dilate(ink, CLEARANCE) & around).any():
                break
        else:
            return None, "in-note sign touches something" if sign["head"] is not None else "signature sign touches something"
        image = np.minimum(image, new[..., None])
        occupied |= ink
    return image, ""


def _write(image: np.ndarray, tokens: str, base: str) -> None:
    rows = read_rows(os.path.join(git_root, tokens))
    with open(base + ".tokens", "w", encoding="utf-8") as handle:
        handle.writelines(" ".join(row) + "\n" for row in rows)
    pixmap = fitz.Pixmap(fitz.csRGB, image.shape[1], image.shape[0], np.ascontiguousarray(image).tobytes(), False)
    pixmap.save(base + ".png")


def more_rare(split: str) -> None:
    """Each staff with a rare sign among its notes, once more in every seen font it has not been set in.

    One font per staff leaves the 1-comma sharp in front of a note some 260
    times across seven fonts, and a sign is read among the notes about as well
    as it is shown there.
    """
    index_path = os.path.join(out_root, f"index_{split}.txt")
    folder = os.path.join(out_root, split)
    existing = [line for line in open(index_path, encoding="utf-8") if line.strip()]
    faces = {name: fitz.Font(fontbuffer=font_buffer(name)) for name in TRAIN_FONTS}
    works = staves_of({"train": index_train, "val": index_val, "test": index_test}[split])
    rel = lambda p: os.path.relpath(p, git_root).replace(os.sep, "/")  # noqa: E731
    added: list[str] = []
    for work in sorted(works):
        chosen = [(n, t) for n, _, t in works[work]
                  if any(r[0].startswith("note") and r[2] in RARE for r in read_rows(os.path.join(git_root, t)))]
        if not chosen:
            continue
        path = os.path.join(symbtr_pdf, work + ".pdf")
        with fitz.open(path) as document:
            if not uses_usual_encoding(document):
                continue
            staves = [(page.number, lines) for page in document for lines in _staff_lines(page)]
            lifts = {n: {s["lift"] for s in signs_of(document[staves[n][0]], staves[n][1])}
                     for n, _ in chosen if n < len(staves)}
        for staff_number, tokens in chosen:
            if not lifts.get(staff_number):
                continue
            page_number, lines = staves[staff_number]
            for font in TRAIN_FONTS:
                base = os.path.join(folder, f"{work}-{staff_number:02d}-f{font}")
                if os.path.exists(base + ".png") or not all(glyph_for(faces[font], l) for l in lifts[staff_number]):
                    continue
                image, _ = redraw(path, page_number, lines, font)
                if image is not None:
                    _write(image, tokens, base)
                    added.append(f"{rel(base + '.png')},{rel(base + '.tokens')}\n")
    with open(index_path, "w", encoding="utf-8") as handle:
        handle.writelines(existing + added)
    eprint(f"{len(added)} more staffs with a rare sign among the notes -> {index_path}")


def merge_parts(split: str, passes: dict[str, tuple[str, ...]]) -> None:
    for group in passes:
        suffix = "" if group == "seen" else "_unseen"
        stem = os.path.join(out_root, f"index_{split}{suffix}")
        parts = sorted(glob.glob(stem + ".part*.txt"))
        lines_out = [line for path in parts for line in open(path, encoding="utf-8") if line.strip()]
        with open(stem + ".txt", "w", encoding="utf-8") as handle:
            handle.writelines(lines_out)
        for path in parts:
            os.remove(path)
        eprint(f"{len(lines_out)} staffs ({group} fonts) from {len(parts)} parts -> {stem}.txt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--limit", type=int, default=None, help="Only N works.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--part", default=None, help="K/N: only every Nth work from the Kth, to run N at once.")
    parser.add_argument("--merge", action="store_true", help="Join the index files the parts wrote.")
    parser.add_argument(
        "--more-rare", action="store_true",
        help="Afterwards: set every staff with a rare sign among its notes in each of the other fonts too.",
    )
    options = parser.parse_args()
    # Training draws on the seen fonts; the test set is set twice, once in
    # those and once in the fonts training never shows.
    passes = {"seen": TRAIN_FONTS} if options.split == "train" else {"seen": TRAIN_FONTS, "unseen": UNSEEN_FONTS}
    if options.merge:
        merge_parts(options.split, passes)
        return
    if options.more_rare:
        more_rare(options.split)
        return

    index = {"train": index_train, "val": index_val, "test": index_test}[options.split]
    works = staves_of(index)
    names = sorted(works)[: options.limit] if options.limit else sorted(works)
    part = ""
    if options.part:
        k, n = (int(v) for v in options.part.split("/"))
        names, part = names[k::n], f".part{k}"
        options.seed += k
    folder = os.path.join(out_root, options.split)
    os.makedirs(folder, exist_ok=True)
    rng = random.Random(options.seed)
    faces = {name: fitz.Font(fontfile=font_path(name)) for group in passes.values() for name in group}

    made: dict[str, list[str]] = {name: [] for name in passes}
    by_font: collections.Counter = collections.Counter()
    by_sign: collections.Counter = collections.Counter()
    tried = 0
    refused: collections.Counter = collections.Counter()
    rel = lambda p: os.path.relpath(p, git_root).replace(os.sep, "/")  # noqa: E731
    for number_of_work, work in enumerate(names):
        path = os.path.join(symbtr_pdf, work + ".pdf")
        with fitz.open(path) as document:
            if not uses_usual_encoding(document):
                continue
            staves = [(page.number, lines) for page in document for lines in _staff_lines(page)]
            lifts = {i: {s["lift"] for s in signs_of(document[p], lines)} for i, (p, lines) in enumerate(staves)}
        for staff_number, _, tokens in works[work]:
            if staff_number >= len(staves) or not lifts[staff_number]:
                continue
            page_number, lines = staves[staff_number]
            for group, fonts in passes.items():
                able = [f for f in fonts if all(glyph_for(faces[f], lift) for lift in lifts[staff_number])]
                if not able:
                    continue
                font = rng.choice(able)
                tried += 1
                image, why = redraw(path, page_number, lines, font)
                if image is None:
                    refused[why] += 1
                    continue
                base = os.path.join(folder, f"{work}-{staff_number:02d}-f{font}")
                _write(image, tokens, base)
                made[group].append(f"{rel(base + '.png')},{rel(base + '.tokens')}\n")
                by_font[font] += 1
                for lift in lifts[staff_number]:
                    by_sign[lift] += 1
        if (number_of_work + 1) % 100 == 0:
            eprint(f"  {number_of_work + 1}/{len(names)} works, {sum(map(len, made.values()))} staffs")

    for group, lines_out in made.items():
        suffix = "" if group == "seen" else "_unseen"
        index_path = os.path.join(out_root, f"index_{options.split}{suffix}{part}.txt")
        with open(index_path, "w", encoding="utf-8") as handle:
            handle.writelines(lines_out)
        eprint(f"{len(lines_out)} staffs ({group} fonts) -> {index_path}")
    eprint(f"{sum(refused.values())} of {tried} staffs left out: "
           + ", ".join(f"{why} {n}" for why, n in refused.most_common()))
    eprint("By font: " + ", ".join(f"{f} {n}" for f, n in by_font.most_common()))
    eprint("Staffs carrying each sign: " + ", ".join(f"{s} {n}" for s, n in by_sign.most_common()))


if __name__ == "__main__":
    main()
