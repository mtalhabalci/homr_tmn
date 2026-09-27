"""Staffs with a double sharp or a double flat in front of one note, to see whether the model reads them.

The makam corpus never draws these two signs, but the vocabulary has them ("##", "bb", kept from homr's
Western training), and Turkish scores do use them now and then. A printed staff is rendered with its signs
redrawn in Bravura (see fonts.py), and one ordinary sharp or ordinary flat in front of a note is drawn as
the double sign instead (SMuFL U+E263 / U+E264). The label of that note gets the lift "##" or "bb"; its
pitch stays, since only the sign changes.

A note's row in the label is found by order: the n-th notehead on the page is the n-th note row. Staffs
where the two counts differ, or where the row's lift is not the sign that was replaced, are left out.

For training the fonts take turns staff by staff over the seven fonts training sees (fonts.TRAIN_FONTS); the
test set is drawn in Bravura alone.

v14 read 38 of 39 such signs as the 4-comma flat and none as a double sign: what homr knew of them was lost in
the makam fine-tuning, which never showed one.

    python -m training.datasets.double_signs --split test --limit 60
    python -m training.datasets.double_signs --split train
"""

import argparse
import os

import fitz
import numpy as np

from homr.simple_logging import eprint
from training.datasets import fonts
from training.datasets.convert_symbtr import _staff_lines, dataset_root, git_root, index_test, index_train, symbtr_pdf
from training.datasets.courtesy import read_rows, staves_of
from training.datasets.page_notes import read_staff, uses_usual_encoding

out_root = os.path.join(dataset_root, "double-signs")
DOUBLE = {"sharp4": ("##", 0xE263), "flat5": ("bb", 0xE264)}
FONT = "Bravura"


def pick(page: fitz.Page, lines: list[float], rows: list[list[str]]) -> tuple[int, str] | None:
    """The first ordinary sharp or flat in front of a note whose row can be found: (row index, lift)."""
    heads = read_staff(page, lines)
    note_rows = [i for i, r in enumerate(rows) if r[0].startswith("note")]
    if len(heads) != len(note_rows):
        return None
    for head, row in zip(heads, note_rows):
        sign = head.get("sign")
        if sign is None:
            continue
        lift = rows[row][2]
        if lift in DOUBLE:
            return row, lift
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--limit", type=int, default=None, help="Works to look through.")
    options = parser.parse_args()
    folder = os.path.join(out_root, options.split)
    os.makedirs(folder, exist_ok=True)
    works = staves_of(index_test if options.split == "test" else index_train)
    turn = (FONT,) if options.split == "test" else fonts.TRAIN_FONTS
    faces = {name: fitz.Font(fontbuffer=fonts.font_buffer(name)) for name in turn}
    original_glyph_for = fonts.glyph_for
    made, rel = [], (lambda p: os.path.relpath(p, git_root).replace(os.sep, "/"))
    for work in sorted(works)[: options.limit]:
        path = os.path.join(symbtr_pdf, work + ".pdf")
        with fitz.open(path) as document:
            if not uses_usual_encoding(document):
                continue
            staves = [(page.number, lines) for page in document for lines in _staff_lines(page)]
            choices = {}
            for number, _, tokens in works[work]:
                if number >= len(staves):
                    continue
                rows = [list(r) for r in read_rows(os.path.join(git_root, tokens))]
                chosen = pick(document[staves[number][0]], staves[number][1], rows)
                if chosen:
                    choices[number] = (tokens, rows, chosen)
        for number, (tokens, rows, (row, lift)) in choices.items():
            name, code = DOUBLE[lift]
            # The redraw sets every sign in Bravura; the one lift of the chosen note is drawn as the double
            # sign. The signature and other notes keep their meaning, so only one sign of this lift may occur.
            if sum(1 for r in rows if r[2] == lift) != 1:
                continue
            font = turn[len(made) % len(turn)]
            if not faces[font].has_glyph(code):
                continue
            fonts.glyph_for = lambda f, l, _l=lift, _c=code: _c if l == _l else original_glyph_for(f, l)
            try:
                image, why = fonts.redraw(path, staves[number][0], staves[number][1], font)
            finally:
                fonts.glyph_for = original_glyph_for
            if image is None:
                continue
            rows[row][2] = name
            tag = "" if options.split == "test" else f"-f{font}"
            base = os.path.join(folder, f"{work}-{number:02d}-{name.replace('#', 'x')}{tag}")
            with open(base + ".tokens", "w", encoding="utf-8") as handle:
                handle.writelines(" ".join(r) + "\n" for r in rows)
            pixmap = fitz.Pixmap(fitz.csRGB, image.shape[1], image.shape[0],
                                 np.ascontiguousarray(image).tobytes(), False)
            pixmap.save(base + ".png")
            made.append(f"{rel(base + '.png')},{rel(base + '.tokens')}\n")
    index = os.path.join(out_root, f"index_{options.split}.txt")
    with open(index, "w", encoding="utf-8") as handle:
        handle.writelines(made)
    eprint(f"{len(made)} staffs with one double sign -> {index}")


if __name__ == "__main__":
    main()
