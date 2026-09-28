"""Staffs with a double sharp or a double flat in front of one note, to see whether the model reads them.

The makam corpus never draws these two signs, but the vocabulary has them ("##", "bb", kept from homr's
Western training), and Turkish scores do use them now and then. A printed staff is rendered with its signs
redrawn in a SMuFL font (see fonts.py), and one ordinary sharp or ordinary flat in front of a note is drawn
as the double sign instead (SMuFL U+E263 / U+E264). The label of that note gets the lift "##" or "bb"; its
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

The labels are the copies segno_labels.py made with the segnos put back (datasets/SymbTr-segno); a staff
with no segno keeps its own label there. The first version took the old labels, and 40 of its 295 staffs
showed a segno the label said was not there -- the very thing v15 was trained out of. A staff whose page
shows a different number of segnos over it than its label holds is left out; of the 2,854 staffs that
offer a sign, none does, and 69 of the 560 drawn carry a segno.

The exam stays the first version's 39 staffs (Bravura alone, from the test works). Its labels had the same
fault: v15 read three segnos there and was charged for inventing them. --split test copies those 39
pictures byte for byte to double-signs-v2/test and gives each the segno label of its staff with the same
"##" or "bb" row; datasets/double-signs is left as it was.

v14 read 38 of 39 such signs as the 4-comma flat and none as a double sign, and v15, trained on the first
version's staffs, still none (the double sharp became sharp1): the fine-tune's start had pushed "##" and
"bb" out of reach together with "#" and "b" (train.seed_makam_accidentals). homr's checkpoint reads them;
they are no longer retired.

    python -m training.datasets.double_signs --split train
    python -m training.datasets.double_signs --split test
"""

import argparse
import collections
import os
import random
import re
import shutil

import fitz
import numpy as np

from homr.simple_logging import eprint
from training.datasets import fonts
from training.datasets.convert_symbtr import _staff_lines, dataset_root, git_root, index_test, symbtr_pdf
from training.datasets.courtesy import MARGIN, read_rows, staves_of
from training.datasets.page_notes import read_staff, uses_usual_encoding
from training.datasets.segno_labels import OUT as SEGNO_ROOT
from training.datasets.segno_labels import SEGNO

out_root = os.path.join(dataset_root, "double-signs-v2")
# The labels with the segnos segno_labels.py gave back; a staff without one keeps its own label there.
segno_index = {split: os.path.join(SEGNO_ROOT, f"index_{split}.txt") for split in ("train", "test")}
first_exam = os.path.join(dataset_root, "double-signs", "index_test.txt")
DOUBLE = {"flat5": ("bb", 0xE264), "sharp4": ("##", 0xE263)}


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


def segnos_on(page: fitz.Page, staves: list[tuple[int, list[float]]], number: int) -> int:
    """How many segnos Mus2 set over staff `number`, each given to a staff the way segno_labels gives it."""
    numbered = [(n, lines) for n, (p, lines) in enumerate(staves) if p == page.number]
    count = 0
    for block in page.get_text("rawdict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                if "Mus2" not in span.get("font", ""):
                    continue
                for char in span.get("chars", []):
                    if ord(char["c"]) != SEGNO:
                        continue
                    origin = char["origin"][1]
                    gap, nearest = min((max(lines[0] - origin, origin - lines[-1], 0.0), n)
                                       for n, lines in numbered)
                    count += gap <= MARGIN and nearest == number
    return count


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


def rel(path: str) -> str:
    return os.path.relpath(path, git_root).replace(os.sep, "/")


def train(options: argparse.Namespace) -> None:
    folder = os.path.join(out_root, "train")
    os.makedirs(folder, exist_ok=True)
    works = staves_of(segno_index["train"])
    turn = fonts.TRAIN_FONTS
    faces = {name: fitz.Font(fontbuffer=fonts.font_buffer(name)) for name in turn}

    offered = []
    unlabelled: collections.Counter = collections.Counter()
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
                if not found:
                    continue
                # A segno the page shows and the label lacks would teach the model to look past it again.
                shown, labelled = segnos_on(page, staves, number), sum(r[0] == "segno" for r in rows)
                if shown != labelled:
                    unlabelled[f"page {shown} segno, label {labelled}"] += 1
                    continue
                lifts = {s["lift"] for s in fonts.signs_of(page, lines)}
                offered.append((work, number, staves[number], rows, found, lifts))
    eprint(f"{len(offered)} staffs offer a sign: "
           + ", ".join(f"{DOUBLE[lift][0]} {sum(lift in o[4] for o in offered)}" for lift in DOUBLE)
           + "; left out, the segnos of page and label differ: "
           + (", ".join(f"{why} {n}" for why, n in unlabelled.most_common()) or "none"))
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
        base = os.path.join(folder, f"{work}-{number:02d}-{name.replace('#', 'x')}-f{font}")
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
    index = os.path.join(out_root, "index_train.txt")
    with open(index, "w", encoding="utf-8") as handle:
        handle.writelines(sorted(made))
    eprint(f"{len(made)} staffs with one double sign -> {index}: "
           + ", ".join(f"{name} {count[name]}" for name, _ in DOUBLE.values()))
    for name, _ in DOUBLE.values():
        eprint(f"  {name} by font: " + ", ".join(f"{f} {by_font[(name, f)]}" for f in turn))
    eprint("Left out: " + ", ".join(f"{why} {n}" for why, n in refused.most_common()))


def exam() -> None:
    """The first version's exam again, the same pictures byte for byte, its labels given their segnos."""
    folder = os.path.join(out_root, "test")
    os.makedirs(folder, exist_ok=True)
    corrected, original = staves_of(segno_index["test"]), staves_of(index_test)
    made: list[str] = []
    gained = 0
    for line in open(first_exam, encoding="utf-8"):
        if not line.strip():
            continue
        image, tokens = line.strip().split(",")
        stem = os.path.basename(tokens)[: -len(".tokens")]
        work, number = re.fullmatch(r"(.*)-(\d\d)-(?:xx|bb)", stem).groups()
        number = int(number)
        old = read_rows(os.path.join(git_root, tokens))
        source = read_rows(os.path.join(git_root, next(t for n, _, t in original[work] if n == number)))
        segno_label = next(t for n, _, t in corrected[work] if n == number)
        rows = [list(r) for r in read_rows(os.path.join(git_root, segno_label))]
        # The one row the first version changed: the n-th note, its lift made "##" or "bb".
        changed = [i for i, (a, b) in enumerate(zip(old, source)) if a != b]
        if len(old) != len(source) or len(changed) != 1 or old[changed[0]][2] not in ("##", "bb"):
            raise ValueError(f"{tokens}: not one lift changed from {len(source)} source rows")
        nth = sum(1 for r in old[: changed[0]] if r[0].startswith("note"))
        row = [i for i, r in enumerate(rows) if r[0].startswith("note")][nth]
        if rows[row] != source[changed[0]]:
            raise ValueError(f"{tokens}: the segno label's note {nth} is not the note that was changed")
        rows[row][2] = old[changed[0]][2]
        if [r for r in rows if r[0] != "segno"] != old:
            raise ValueError(f"{tokens}: the new label differs from the old by more than segnos")
        with fitz.open(os.path.join(symbtr_pdf, work + ".pdf")) as document:
            staves = [(page.number, lines) for page in document for lines in _staff_lines(page)]
            shown = segnos_on(document[staves[number][0]], staves, number)
        labelled = sum(r[0] == "segno" for r in rows)
        if shown != labelled:
            eprint(f"  {stem}: the page shows {shown} segno, the label has {labelled}")
        gained += labelled > 0
        base = os.path.join(folder, stem)
        shutil.copyfile(os.path.join(git_root, image), base + ".png")
        with open(base + ".tokens", "w", encoding="utf-8") as handle:
            handle.writelines(" ".join(r) + "\n" for r in rows)
        made.append(f"{rel(base + '.png')},{rel(base + '.tokens')}\n")
    index = os.path.join(out_root, "index_test.txt")
    with open(index, "w", encoding="utf-8") as handle:
        handle.writelines(made)
    eprint(f"{len(made)} exam staffs, {gained} of them given a segno -> {index}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--limit", type=int, default=None, help="Works to look through.")
    parser.add_argument("--most", type=int, default=280, help="Staffs for each double sign at most.")
    parser.add_argument("--seed", type=int, default=0)
    options = parser.parse_args()
    if options.split == "test":
        exam()
    else:
        train(options)


if __name__ == "__main__":
    main()
