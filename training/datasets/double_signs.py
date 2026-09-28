"""Staffs with a double sharp or a double flat in front of one note, to see whether the model reads them.

The makam corpus never draws these two signs, but the vocabulary has them ("##", "bb", kept from homr's
Western training), and Turkish scores do use them now and then. A printed staff is rendered with its signs
redrawn in Bravura (see fonts.py), and one ordinary sharp or ordinary flat in front of a note is drawn as
the double sign instead (SMuFL U+E263 / U+E264). The label of that note gets the lift "##" or "bb"; its
pitch stays, since only the sign changes.

A note's row in the label is found by order: the n-th notehead on the page is the n-th note row. Staffs
where the two counts differ are left out, and so is a note whose row does not have the lift of the sign
in front of it.

Only that one sign becomes the double sign. The first version redrew every sign with the chosen lift, so it
could only use staffs where that lift stood once, key signature included; the flat nearly always stands in
the key signature, and training got 280 double sharps and 15 double flats. Now the sign read_staff finds in
front of the chosen note is marked by its place on the page when fonts.redraw takes the signs off, and only
the marked sign is drawn as the double sign: the key signature and the other notes keep their own signs and
their own rows. Of the training staffs 611 offer a flat in front of a note and 2,391 a sharp. A staff that
offers both gives the flat, and each double sign is capped at 280 staffs (--most), the staffs taken in a
seeded random order so the cap does not fall on the makams early in the alphabet.

For training the fonts take turns over the seven fonts training sees (fonts.TRAIN_FONTS), for each double
sign on its own, but a staff only goes to a font that draws every sign on it. Leland, MuseJazz and
FinaleBroadway lack the 1-comma flat most makam key signatures carry; with the strict turn the first
version had, a flat whose turn fell to one of them was mostly left out (171 double flats, not 280). Now
those three get about 20 double flats each and the other four about 55.

The exam stays the first version's test split, datasets/double-signs/index_test.txt (39 staffs, Bravura
alone, from the test works); this version writes to double-signs-v2 and leaves it alone.

v14 read 38 of 39 such signs as the 4-comma flat and none as a double sign, and v15, trained on the first
version's staffs, still none (the double sharp became sharp1): the fine-tune's start had pushed "##" and
"bb" out of reach together with "#" and "b" (train.seed_makam_accidentals). homr's checkpoint reads them;
they are no longer retired.

    python -m training.datasets.double_signs --split train
"""

import argparse
import collections
import os
import random

import fitz
import numpy as np

from homr.simple_logging import eprint
from training.datasets import fonts
from training.datasets.convert_symbtr import _staff_lines, dataset_root, git_root, index_test, index_train, symbtr_pdf
from training.datasets.courtesy import read_rows, staves_of
from training.datasets.page_notes import read_staff, uses_usual_encoding

out_root = os.path.join(dataset_root, "double-signs-v2")
DOUBLE = {"flat5": ("bb", 0xE264), "sharp4": ("##", 0xE263)}
FONT = "Bravura"


def candidates(page: fitz.Page, lines: list[float], rows: list[list[str]]) -> dict[str, tuple[int, dict]]:
    """For each ordinary sign, the first note it stands in front of whose row has it too.

    lift -> (row index, the sign as read_staff found it)
    """
    heads = read_staff(page, lines)
    note_rows = [i for i, r in enumerate(rows) if r[0].startswith("note")]
    if len(heads) != len(note_rows):
        return {}
    found: dict[str, tuple[int, dict]] = {}
    for head, row in zip(heads, note_rows):
        lift = head["lift"]
        if head["sign"] is not None and lift in DOUBLE and rows[row][2] == lift and lift not in found:
            found[lift] = (row, head["sign"])
    return found


def redraw_one(path: str, page_number: int, lines: list[float], font: str, target: dict, name: str,
               ) -> tuple[np.ndarray | None, str]:
    """The staff redrawn in the font with only the target sign drawn as the double sign `name`.

    fonts.redraw looks each sign's glyph up by its lift. The target is found among the signs it takes off
    the page by the same test redraw uses to tie a sign to its notehead (code and place), and given the lift
    "##" or "bb", which only the double glyph answers to.
    """
    original_signs_of, original_glyph_for = fonts.signs_of, fonts.glyph_for
    code = dict(DOUBLE.values())[name]
    marked: list[dict] = []

    def signs_of(page: fitz.Page, staff: list[float]) -> list[dict]:
        found = original_signs_of(page, staff)
        for sign in found:
            if fonts._same(sign, target):
                sign["lift"] = name
                marked.append(sign)
        return found

    def glyph_for(face: fitz.Font, lift: str) -> int | None:
        if lift == name:
            return code if face.has_glyph(code) else None
        return original_glyph_for(face, lift)

    fonts.signs_of, fonts.glyph_for = signs_of, glyph_for
    try:
        image, why = fonts.redraw(path, page_number, lines, font)
    finally:
        fonts.signs_of, fonts.glyph_for = original_signs_of, original_glyph_for
    if image is not None and not marked:
        return None, "chosen sign not among the redrawn ones"
    return image, why


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--limit", type=int, default=None, help="Works to look through.")
    parser.add_argument("--most", type=int, default=280, help="Staffs for each double sign at most.")
    parser.add_argument("--seed", type=int, default=0)
    options = parser.parse_args()
    folder = os.path.join(out_root, options.split)
    os.makedirs(folder, exist_ok=True)
    works = staves_of(index_test if options.split == "test" else index_train)
    turn = (FONT,) if options.split == "test" else fonts.TRAIN_FONTS
    faces = {name: fitz.Font(fontbuffer=fonts.font_buffer(name)) for name in turn}
    rel = lambda p: os.path.relpath(p, git_root).replace(os.sep, "/")  # noqa: E731

    offered = []
    for work in sorted(works)[: options.limit]:
        path = os.path.join(symbtr_pdf, work + ".pdf")
        with fitz.open(path) as document:
            if not uses_usual_encoding(document):
                continue
            staves = [(page.number, lines) for page in document for lines in _staff_lines(page)]
            for number, _, tokens in works[work]:
                if number >= len(staves):
                    continue
                page, lines = document[staves[number][0]], staves[number][1]
                rows = [list(r) for r in read_rows(os.path.join(git_root, tokens))]
                found = candidates(page, lines, rows)
                if found:
                    lifts = {s["lift"] for s in fonts.signs_of(page, lines)}
                    offered.append((work, number, staves[number], rows, found, lifts))
    eprint(f"{len(offered)} staffs offer a sign: "
           + ", ".join(f"{DOUBLE[lift][0]} {sum(lift in o[4] for o in offered)}" for lift in DOUBLE))
    random.Random(options.seed).shuffle(offered)

    made: list[str] = []
    count: collections.Counter = collections.Counter()
    by_font: collections.Counter = collections.Counter()
    refused: collections.Counter = collections.Counter()
    for work, number, (page_number, lines), rows, found, lifts in offered:
        path = os.path.join(symbtr_pdf, work + ".pdf")
        drawn = None
        # The flat first: the staffs that have one are the scarcer.
        for lift, (name, code) in DOUBLE.items():
            if lift not in found or count[name] >= options.most:
                continue
            row, sign = found[lift]
            # Three fonts lack the 1-comma flat most key signatures carry: the turn goes round the fonts
            # that draw every sign on the staff, the one this double sign has had least first.
            able = [f for f in turn
                    if faces[f].has_glyph(code) and all(fonts.glyph_for(faces[f], l) for l in lifts)]
            why = "no font draws every sign"
            for font in sorted(able, key=lambda f: by_font[(name, f)]):
                image, why = redraw_one(path, page_number, lines, font, sign, name)
                if image is not None:
                    drawn = image, row, name, font
                    break
            if drawn:
                break
            refused[f"{name}: {why}"] += 1
        if drawn is None:
            continue
        image, row, name, font = drawn
        labelled = [list(r) for r in rows]
        labelled[row][2] = name
        tag = "" if options.split == "test" else f"-f{font}"
        base = os.path.join(folder, f"{work}-{number:02d}-{name.replace('#', 'x')}{tag}")
        with open(base + ".tokens", "w", encoding="utf-8") as handle:
            handle.writelines(" ".join(r) + "\n" for r in labelled)
        pixmap = fitz.Pixmap(fitz.csRGB, image.shape[1], image.shape[0],
                             np.ascontiguousarray(image).tobytes(), False)
        pixmap.save(base + ".png")
        made.append(f"{rel(base + '.png')},{rel(base + '.tokens')}\n")
        count[name] += 1
        by_font[(name, font)] += 1
        if all(count[name] >= options.most for name, _ in DOUBLE.values()):
            break
    index = os.path.join(out_root, f"index_{options.split}.txt")
    with open(index, "w", encoding="utf-8") as handle:
        handle.writelines(sorted(made))
    eprint(f"{len(made)} staffs with one double sign -> {index}: "
           + ", ".join(f"{name} {count[name]}" for name, _ in DOUBLE.values()))
    for name, _ in DOUBLE.values():
        eprint(f"  {name} by font: " + ", ".join(f"{f} {by_font[(name, f)]}" for f in turn))
    eprint("Left out: " + ", ".join(f"{why} {n}" for why, n in refused.most_common()))


if __name__ == "__main__":
    main()
