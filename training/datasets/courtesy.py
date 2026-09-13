"""New in-note examples of the scarce accidentals, drawn in Mus2's own hand.

Every class of sign reads at essentially 100% in the key signature and falls
away among the notes in step with how often training shows it there: the
five-comma sharp stands in front of a note only 69 times in the whole training
set. Repeating those few staffs teaches the staffs, not the sign.

A courtesy accidental is the musical way out. A note whose letter the key
signature already alters -- every F in a mahur piece -- may carry that same
sign again in front of it; the engraver restates what is in force and nothing
sounds different. So for every such note there is a correct page with the sign
drawn in front of it, and a correct label: the same note, now carrying the sign.

This draws that page. The staff is rendered exactly as the dataset renders it;
the sign is rendered alone, from the font embedded in the same pdf, at the size
and distance from the notehead Mus2 itself uses (measured over the corpus, see
OFFSET); the two are laid over each other. A sign that would touch anything but
the staff lines -- a stem, a beam, a slur, the note before -- is not drawn: Mus2
would have made room for it, and we cannot move the notes.

The 2-comma sharp is never in a key signature, so there is nothing to restate.
For it the sign is simply added in front of a free note: the picture and the
label still agree exactly, only the melody moves a comma, which the reader of
shapes does not care about. Its glyph comes from a pdf that uses it, since the
font is embedded per pdf with only the characters that pdf needs.

    python -m training.datasets.courtesy --split train
    python -m training.datasets.courtesy --split test
"""

import argparse
import collections
import os
import random
import re

import fitz
import numpy as np

from homr.simple_logging import eprint
from training.datasets.convert_symbtr import (
    ACCIDENTAL_GLYPHS,
    _staff_lines,
    dataset_root,
    git_root,
    index_test,
    index_train,
    index_val,
    symbtr_pdf,
    target_page_width,
)
from training.datasets.page_notes import NATURAL, pair_notes, read_staff, uses_usual_encoding

# The signs to restate. The common ones are left alone: they already appear in
# front of notes thousands of times.
TARGETS = ("sharp5", "sharp3", "sharp1", "flat2", "flat3")
# Signs no key signature carries, added in front of a free note instead.
FREE = ("sharp2",)
# Letters a free sharp is put on: the ones the corpus sharpens.
FREE_LETTERS = "FCG"

CODE_OF = {
    "N": NATURAL,
    **{(f"sharp{c}" if c > 0 else f"flat{-c}"): code for code, c in ACCIDENTAL_GLYPHS.items()},
}
# Distance from the sign's origin to the notehead's origin, as a share of the
# font size. Mus2 sets it by glyph and nothing else: over 4,253 signs on 400
# works the tenth and ninetieth percentiles all but coincide with the median.
OFFSET = {
    0x6E: 0.225, 0x58: 0.260, 0x51: 0.300, 0x54: 0.275, 0x50: 0.215, 0x55: 0.195,
    0x53: 0.300, 0x57: 0.340, 0x52: 0.314, 0x59: 0.235, 0x56: 0.240, 0x5C: 0.260,
}
MARGIN = 34
ENDS = ("barline", "repeat", "volta", "bolddoublebarline")
STAFF = re.compile(r"-(\d+)$")
# Ink darker than this counts as something drawn.
INK = 200
# Room, in pixels, a sign keeps from everything around it.
CLEARANCE = 2

out_root = os.path.join(dataset_root, "SymbTr-courtesy")


def read_rows(path: str) -> list[list[str]]:
    return [line.split() for line in open(path, encoding="utf-8") if len(line.split()) == 5]


def courtesy_candidates(rows: list[list[str]], memory: dict[str, str]) -> list[tuple[int, str]]:
    """(note number on the staff, sign) for each note a sign could be restated on.

    memory is the measure's signs so far, carried in from the previous staff
    when Mus2 broke the measure across the line, and updated here.
    """
    signature: dict[str, str] = {}
    found = []
    number = -1
    for rhythm, pitch, lift, articulation, _ in rows:
        if rhythm == "keyAccidental":
            signature[pitch[0]] = lift
            continue
        if rhythm.startswith(ENDS):
            memory.clear()
            continue
        if not rhythm.startswith("note"):
            continue
        number += 1
        if lift not in ("_", "."):
            memory[pitch] = lift
            continue
        if rhythm.endswith("G") or "tieStop" in articulation:
            continue
        sign = memory.get(pitch) or signature.get(pitch[0])
        if sign in TARGETS:
            found.append((number, sign))
    return found


def free_candidates(rows: list[list[str]]) -> list[tuple[int, str]]:
    found, number = [], -1
    for rhythm, pitch, lift, articulation, _ in rows:
        if not rhythm.startswith("note"):
            continue
        number += 1
        if lift in ("_", ".") and not rhythm.endswith("G") and "tieStop" not in articulation:
            if pitch[0] in FREE_LETTERS:
                found.extend((number, sign) for sign in FREE)
    return found


def _mus2_fonts(document: fitz.Document) -> list[bytes]:
    """The pdf's copies of Mus2 that follow the usual codes.

    The CID-keyed copy some pdfs also carry numbers its signs differently, so
    asking it for a code could draw a different sign from the one labelled.
    """
    buffers = []
    for page in document:
        for xref, _, _, base, _, _, _ in page.get_fonts(full=True):
            if "Mus2" in base and "Identity" not in base:
                buffer = document.extract_font(xref)[3]
                if buffer and buffer not in buffers:
                    buffers.append(buffer)
    return buffers


def font_with(buffers: list[bytes], code: int) -> bytes | None:
    for buffer in buffers:
        if fitz.Font(fontbuffer=buffer).has_glyph(code):
            return buffer
    return None


def _pixels(page: fitz.Page, clip: fitz.Rect, dpi: int) -> np.ndarray:
    pixmap = page.get_pixmap(clip=clip, dpi=dpi)
    return np.frombuffer(pixmap.samples, np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n)[..., :3]


def sign_pixels(
    size: tuple[float, float], clip: fitz.Rect, dpi: int, buffer: bytes, code: int,
    origin: tuple[float, float], font_size: float,
) -> np.ndarray:
    """The sign alone on an empty page of the same size, rendered like the staff."""
    blank = fitz.open()
    page = blank.new_page(width=size[0], height=size[1])
    page.insert_font(fontname="sign", fontbuffer=buffer)
    page.insert_text(fitz.Point(*origin), chr(code), fontname="sign", fontsize=font_size, color=(0, 0, 0))
    pixels = _pixels(page, clip, dpi)
    blank.close()
    return pixels


def _dilate(mask: np.ndarray, by: int) -> np.ndarray:
    grown = mask.copy()
    for dy in range(-by, by + 1):
        for dx in range(-by, by + 1):
            grown |= np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
    return grown


def draw_signs(
    page: fitz.Page, lines: list[float], heads: list[dict], wanted: list[tuple[int, str]],
    fonts: list[bytes], donors: dict[int, bytes], limit: int,
) -> tuple[np.ndarray, list[tuple[int, str]]] | None:
    """The staff image with as many of the wanted signs as fit cleanly, up to limit."""
    clip = fitz.Rect(0, lines[0] - MARGIN, page.rect.width, lines[-1] + MARGIN)
    dpi = round(target_page_width / page.rect.width * 72)
    image = _pixels(page, clip, dpi).copy()
    scale = dpi / 72
    # Everything already on the staff except the staff lines themselves.
    occupied = image.min(axis=2) < INK
    for y in lines:
        row = int(round((y - clip.y0) * scale))
        occupied[max(row - 2, 0) : row + 3, :] = False
    drawn = []
    for head_number, sign in wanted:
        if len(drawn) >= limit:
            break
        head = heads[head_number]
        code = CODE_OF[sign]
        buffer = font_with(fonts, code) or donors.get(code)
        if buffer is None:
            continue
        size = head["size"]
        origin = (head["origin"][0] - OFFSET[code] * size, head["origin"][1])
        alone = sign_pixels((page.rect.width, page.rect.height), clip, dpi, buffer, code, origin, size)
        ink = alone.min(axis=2) < INK
        # The sign sits as close to its own notehead as Mus2 sets it, so only
        # what lies left of that notehead can be in the way.
        around = occupied.copy()
        around[:, max(int((head["x"] - clip.x0) * scale) - 1, 0) :] = False
        if not ink.any() or (_dilate(ink, CLEARANCE) & around).any():
            continue
        image = np.minimum(image, alone)
        occupied |= ink
        drawn.append((head_number, sign))
    if not drawn:
        return None
    return image, drawn


def staves_of(index: str) -> dict[str, list[tuple[int, str, str]]]:
    works: dict[str, list] = collections.defaultdict(list)
    seen = set()
    for line in open(index, encoding="utf-8"):
        if not line.strip() or line in seen:
            continue
        seen.add(line)
        image, tokens = line.strip().split(",")
        name = os.path.basename(tokens)[: -len(".tokens")]
        match = STAFF.search(name)
        works[name[: match.start()]].append((int(match.group(1)), image, tokens))
    return {work: sorted(staves) for work, staves in works.items()}


def find_donors() -> dict[int, bytes]:
    """A font for each sign that is rare enough to be missing from most pdfs."""
    donors: dict[int, bytes] = {}
    wanted = {CODE_OF[sign] for sign in FREE}
    for index in (index_train, index_val, index_test):
        for work in staves_of(index):
            if wanted <= set(donors):
                return donors
            path = os.path.join(symbtr_pdf, work + ".pdf")
            with fitz.open(path) as document:
                for code in wanted - set(donors):
                    buffer = font_with(_mus2_fonts(document), code)
                    if buffer is not None:
                        donors[code] = buffer
                        eprint(f"Glyph {code:#04x} taken from {work}")
    return donors


def write_staff(image: np.ndarray, rows: list[list[str]], drawn: list[tuple[int, str]], base: str) -> None:
    signs = dict(drawn)
    number = -1
    with open(base + ".tokens", "w", encoding="utf-8") as handle:
        for row in rows:
            if row[0].startswith("note"):
                number += 1
                if number in signs:
                    row = [row[0], row[1], signs[number], row[3], row[4]]
            handle.write(" ".join(row) + "\n")
    pixmap = fitz.Pixmap(fitz.csRGB, image.shape[1], image.shape[0], np.ascontiguousarray(image).tobytes(), False)
    pixmap.save(base + ".png")


def main() -> None:  # noqa: PLR0915
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--per-staff", type=int, default=3, help="Most signs added to one staff.")
    parser.add_argument("--cap", type=int, default=1500, help="Most signs of one kind in all.")
    parser.add_argument("--limit", type=int, default=None, help="Only N works.")
    parser.add_argument("--seed", type=int, default=0)
    options = parser.parse_args()

    index = {"train": index_train, "val": index_val, "test": index_test}[options.split]
    works = staves_of(index)
    names = sorted(works)[: options.limit] if options.limit else sorted(works)
    folder = os.path.join(out_root, options.split)
    os.makedirs(folder, exist_ok=True)
    rng = random.Random(options.seed)
    donors = find_donors()

    added: collections.Counter = collections.Counter()
    offered: collections.Counter = collections.Counter()
    lines_out = []
    for number_of_work, work in enumerate(names):
        path = os.path.join(symbtr_pdf, work + ".pdf")
        with fitz.open(path) as document:
            if not uses_usual_encoding(document):
                continue
            fonts = _mus2_fonts(document)
            staves = [(page, lines) for page in document for lines in _staff_lines(page)]
            memory: dict[str, str] = {}
            for staff_number, _, tokens in works[work]:
                rows = read_rows(os.path.join(git_root, tokens))
                restated = courtesy_candidates(rows, memory)
                if staff_number >= len(staves):
                    continue
                page, lines = staves[staff_number]
                heads = read_staff(page, lines)
                where = pair_notes([(r[0], r[1]) for r in rows if r[0].startswith("note")], heads)
                if where is None:
                    continue
                free = free_candidates(rows)
                for _, sign in restated + free:
                    offered[sign] += 1
                # Restated signs first; a free one only to top a staff up, one
                # at most, or it would crowd out the rest -- every F is a candidate.
                rng.shuffle(restated)
                rng.shuffle(free)
                wanted = [
                    (where[n], sign)
                    for n, sign in restated + free[:1]
                    if where[n] is not None and added[sign] < options.cap
                ]
                result = draw_signs(page, lines, heads, wanted, fonts, donors, options.per_staff)
                if result is None:
                    continue
                image, drawn = result
                # draw_signs speaks in heads; the labels speak in note numbers.
                note_of_head = {w: n for n, w in enumerate(where) if w is not None}
                drawn_notes = [(note_of_head[h], sign) for h, sign in drawn]
                base = os.path.join(folder, f"{work}-{staff_number:02d}-c")
                write_staff(image, rows, drawn_notes, base)
                for _, sign in drawn_notes:
                    added[sign] += 1
                rel = lambda p: os.path.relpath(p, git_root).replace(os.sep, "/")  # noqa: E731
                lines_out.append(f"{rel(base + '.png')},{rel(base + '.tokens')}\n")
        if (number_of_work + 1) % 100 == 0:
            eprint(f"  {number_of_work + 1}/{len(names)} works, {len(lines_out)} staffs")

    index_path = os.path.join(out_root, f"index_{options.split}.txt")
    with open(index_path, "w", encoding="utf-8") as handle:
        handle.writelines(lines_out)
    eprint(f"\n{len(lines_out)} staffs with signs added -> {index_path}")
    eprint(f"{'sign':<8}{'could go':>10}{'drawn':>8}")
    for sign in (*TARGETS, *FREE):
        eprint(f"{sign:<8}{offered[sign]:>10}{added[sign]:>8}")


if __name__ == "__main__":
    main()
