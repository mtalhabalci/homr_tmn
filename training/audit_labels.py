"""Check every label against the page it describes, one note at a time.

The labels are not read off the page. They are worked out from the .mu2 note
list and the rules of engraving: the key signature, the measure's memory. Where
the pdf was engraved from exactly that list, the two agree. Where it was not --
a later edition, a hand correction -- the page can carry a sign the label does
not know about, and the model is then taught to look past it, and marked wrong
in the test when it does not.

Totals per class cannot see this: a sign missing here and an extra one there
cancel out. This pairs each note token with its notehead on the page, checks
the pairing by height (the notehead's staff degree must equal the token's
pitch), and then compares the sign drawn just left of the notehead with the one
in the label.

    python -m training.audit_labels
"""

import argparse
import collections
import os
import re

import fitz

from homr.simple_logging import eprint
from training.datasets.convert_symbtr import (
    ACCIDENTAL_GLYPHS,
    NOTEHEAD_GLYPHS,
    _staff_lines,
    git_root,
    index_test,
    index_train,
    index_val,
    symbtr_pdf,
)
from training.datasets.shift_staff import MARGIN, _characters, _degree

NATURAL = 0x6E
LETTERS = "CDEFGAB"
STAFF = re.compile(r"-(\d+)$")

# How far left of its notehead an accidental may start, in staff steps.
REACH = 5
# How far above or below the staff a notehead may sit, in staff steps.
LEDGER_REACH = 9


def lift_of_glyph(code: int) -> str | None:
    if code == NATURAL:
        return "N"
    commas = ACCIDENTAL_GLYPHS.get(code)
    if commas is None:
        return None
    return f"sharp{commas}" if commas > 0 else f"flat{-commas}"


def pitch_at(y: float, bottom: float, step: float) -> str:
    degree = round((bottom - y) / step) + _degree("E4")
    return f"{LETTERS[degree % 7]}{degree // 7}"


def read_staff(page: "fitz.Page", staff: list[float]) -> list[dict]:  # noqa: F821
    """The noteheads of one staff, left to right, each with the sign before it.

    A Mus2 notehead glyph is anchored on its staff degree, so its origin gives
    the pitch directly. An accidental is anchored the same way, on the note it
    belongs to, and stands just to the left.
    """
    top, bottom = staff[0], staff[-1]
    step = (bottom - top) / 8
    band = fitz.Rect(0, top - MARGIN, page.rect.width, bottom + MARGIN)
    glyphs = [g for g in _characters(page, band) if g["notation"]]
    heads = []
    for glyph in glyphs:
        if glyph["code"] not in NOTEHEAD_GLYPHS:
            continue
        y = glyph["origin"][1]
        # Further out than four ledger lines is not this staff's: the band also
        # catches the small notes some scores print beneath it for the usul.
        if not top - LEDGER_REACH * step <= y <= bottom + LEDGER_REACH * step:
            continue
        # Mus2 builds some notes from two glyphs laid on the same spot -- a head
        # and the flag or stem that goes with it. One note, not two.
        if heads and abs(heads[-1]["x"] - glyph["x"]) < 1 and abs(heads[-1]["origin"][1] - y) < 1:
            continue
        heads.append(glyph)
    if not heads:
        return []
    first = heads[0]["x"]
    signs = [
        g for g in glyphs if lift_of_glyph(g["code"]) and g["x"] >= first - REACH * step
    ]
    notes = []
    for head in heads:
        y = head["origin"][1]
        before = [
            s
            for s in signs
            if head["x"] - REACH * step <= s["x"] < head["x"]
            and abs(s["origin"][1] - y) < step / 2
        ]
        sign = max(before, key=lambda s: s["x"]) if before else None
        notes.append(
            {
                "x": head["x"],
                "y": y,
                "pitch": pitch_at(y, bottom, step),
                "size": head["size"],
                "code": head["code"],
                "lift": lift_of_glyph(sign["code"]) if sign else None,
                "sign": sign,
            }
        )
    return notes


def note_tokens(path: str) -> list[list[str]]:
    rows = []
    for line in open(path, encoding="utf-8"):
        parts = line.split()
        if len(parts) == 5 and parts[0].startswith("note"):
            rows.append(parts)
    return rows


def _matches(rows: list[list[str]], heads: list[dict]) -> bool:
    return bool(rows) and len(rows) == len(heads) and all(
        row[1] == head["pitch"] for row, head in zip(rows, heads)
    )


def without_grace(rows: list[list[str]], heads: list[dict]) -> tuple[list, list]:
    """Only the full-size notes: a grace note is printed small, and not always
    with a glyph this reads as a notehead."""
    full = max((head["size"] for head in heads), default=0)
    return (
        [row for row in rows if not row[0].endswith("G")],
        [head for head in heads if head["size"] >= 0.85 * full],
    )


def pair_staff(page: "fitz.Page", staff: list[float], tokens: str) -> list[tuple] | None:  # noqa: F821
    """Each note token with its notehead, or None if they cannot be lined up."""
    rows = note_tokens(tokens)
    heads = read_staff(page, staff)
    if _matches(rows, heads):
        return list(zip(rows, heads))
    rows, heads = without_grace(rows, heads)
    if _matches(rows, heads):
        return list(zip(rows, heads))
    return None


def notes_agree_across(staves: list[tuple["fitz.Page", list[float], str]]) -> bool:  # noqa: F821
    """Do the work's notes, taken end to end, match its noteheads end to end?

    When they do and a staff still fails on its own, the notes are right but
    some were filed under the neighbouring staff.
    """
    all_rows, all_heads = [], []
    for page, lines, tokens in staves:
        rows, heads = without_grace(note_tokens(tokens), read_staff(page, lines))
        all_rows += rows
        all_heads += heads
    return _matches(all_rows, all_heads)


def staves_by_work(indexes: list[str]) -> dict[str, list[tuple[int, str]]]:
    works: dict[str, set] = collections.defaultdict(set)
    for index in indexes:
        for line in open(index, encoding="utf-8"):
            if not line.strip():
                continue
            tokens = line.strip().split(",")[1]
            name = os.path.basename(tokens)[: -len(".tokens")]
            match = STAFF.search(name)
            works[name[: match.start()]].add((int(match.group(1)), tokens))
    return {work: sorted(staves) for work, staves in works.items()}


def main() -> None:  # noqa: PLR0915
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Only N works.")
    parser.add_argument("--split", choices=("all", "train", "val", "test"), default="all")
    options = parser.parse_args()

    chosen = {"train": [index_train], "val": [index_val], "test": [index_test]}.get(
        options.split, [index_train, index_val, index_test]
    )
    works = staves_by_work(chosen)
    names = sorted(works)[: options.limit] if options.limit else sorted(works)
    eprint(f"Auditing {len(names)} works")

    staves = paired = 0
    notes = 0
    verdicts: collections.Counter = collections.Counter()
    by_work: collections.Counter = collections.Counter()
    unpaired: collections.Counter = collections.Counter()
    shifted_works: collections.Counter = collections.Counter()
    examples = []
    for work in names:
        path = os.path.join(symbtr_pdf, work + ".pdf")
        if not os.path.exists(path):
            continue
        with fitz.open(path) as document:
            all_staves = [(page, lines) for page in document for lines in _staff_lines(page)]
            mine = [
                (*all_staves[number], os.path.join(git_root, tokens))
                for number, tokens in works[work]
                if number < len(all_staves)
            ]
            failed = 0
            for number, tokens in works[work]:
                staves += 1
                if number >= len(all_staves):
                    unpaired["no such staff on the page"] += 1
                    continue
                page, lines = all_staves[number]
                pairs = pair_staff(page, lines, os.path.join(git_root, tokens))
                if pairs is None:
                    failed += 1
                    continue
                paired += 1
                for row, head in pairs:
                    notes += 1
                    label = row[2] if row[2] not in ("_", ".") else None
                    drawn = head["lift"]
                    if label == drawn:
                        verdicts["agree"] += 1
                        continue
                    if label is None:
                        kind = f"page has {drawn}, label none"
                    elif drawn is None:
                        kind = f"label has {label}, page none"
                    else:
                        kind = f"page {drawn}, label {label}"
                    verdicts[kind] += 1
                    by_work[work] += 1
                    if len(examples) < 15:
                        examples.append(f"{work}-{number:02d} {row[1]}: {kind}")
            if failed:
                if notes_agree_across(mine):
                    unpaired["right notes, filed under a neighbouring staff"] += failed
                    shifted_works[work] += failed
                else:
                    unpaired["notes differ from the page"] += failed

    eprint(f"\nstaves: {staves}, lined up note for note: {paired} = {100 * paired / max(staves, 1):.1f}%")
    for reason, count in unpaired.most_common():
        eprint(f"  {count:>6}  {reason}")
    if shifted_works:
        eprint(f"  works with staves filed wrongly: {len(shifted_works)}")
        for work, count in shifted_works.most_common(8):
            eprint(f"     {count:>4}  {work}")
    eprint(f"notes checked: {notes}")
    wrong = notes - verdicts["agree"]
    eprint(f"  label and page agree : {verdicts['agree']}")
    eprint(f"  they disagree        : {wrong} = {100 * wrong / max(notes, 1):.2f}%")
    for kind, count in verdicts.most_common():
        if kind != "agree":
            eprint(f"     {count:>6}  {kind}")
    eprint(f"\nworks with any disagreement: {len(by_work)}")
    for work, count in by_work.most_common(10):
        eprint(f"   {count:>5}  {work}")
    eprint("\nexamples:")
    for example in examples:
        eprint(f"   {example}")


if __name__ == "__main__":
    main()
