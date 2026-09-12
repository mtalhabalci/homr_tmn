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
in the label. Signs are only compared in pdfs that use the usual Mus2 codes;
see page_notes.uses_usual_encoding.

    python -m training.audit_labels
"""

import argparse
import collections
import os
import re

import fitz

from homr.simple_logging import eprint
from training.datasets.convert_symbtr import (
    _staff_lines,
    git_root,
    index_test,
    index_train,
    index_val,
    symbtr_pdf,
)
from training.datasets.page_notes import full_size, pair_notes, read_staff, uses_usual_encoding

STAFF = re.compile(r"-(\d+)$")


def note_tokens(path: str) -> list[list[str]]:
    rows = []
    for line in open(path, encoding="utf-8"):
        parts = line.split()
        if len(parts) == 5 and parts[0].startswith("note"):
            rows.append(parts)
    return rows


def pair_staff(page: "fitz.Page", staff: list[float], tokens: str) -> list[tuple] | None:  # noqa: F821
    """Each note token with its notehead, or None if they cannot be lined up.

    Grace notes the page does not print as noteheads are left out.
    """
    rows = note_tokens(tokens)
    heads = read_staff(page, staff)
    where = pair_notes([(row[0], row[1]) for row in rows], heads)
    if where is None:
        return None
    return [(row, heads[w]) for row, w in zip(rows, where) if w is not None]


def notes_agree_across(staves: list[tuple["fitz.Page", list[float], str]]) -> bool:  # noqa: F821
    """Do the work's notes, taken end to end, match its noteheads end to end?

    When they do and a staff still fails on its own, the notes are right but
    some were filed under the neighbouring staff.
    """
    all_rows, all_heads = [], []
    for page, lines, tokens in staves:
        all_rows += [r for r in note_tokens(tokens) if not r[0].endswith("G")]
        all_heads += full_size(read_staff(page, lines))
    return bool(all_rows) and len(all_rows) == len(all_heads) and all(
        row[1] == head["pitch"] for row, head in zip(all_rows, all_heads)
    )


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
    unusual_works = 0
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
            usual = uses_usual_encoding(document)
            unusual_works += not usual
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
                if not usual:
                    continue
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
    eprint(f"\nsigns compared in {len(names) - unusual_works} works with the usual Mus2 codes"
           f" ({unusual_works} others left out)")
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
