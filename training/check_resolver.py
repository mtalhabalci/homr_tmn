"""Can the sounding pitch be recovered from what the page prints?

The training labels record engraving: the symbols a Mus2 page actually shows,
with four of every five alterations left to the key signature and the measure's
memory. homr.makam_key puts those back. This runs the round trip over the whole
corpus -- .mu2 to printed symbols to sounding pitch again -- and reports where
the original does not come back.

    python -m training.check_resolver
    python -m training.check_resolver --limit 200

The failures are the interesting part. Rounding is the one loss built into the
encoding: a work that engraves the 2-comma flat with the 1-comma sign leaves no
way to tell them apart afterwards, so those notes cannot come back and the
report separates them from real mistakes.
"""

import argparse
import collections
import os
import re

from homr.makam_key import commas_for_lift, resolve_sounding
from homr.simple_logging import eprint
from homr.transformer.vocabulary import EncodedSymbol
from training.datasets.convert_symbtr import (
    SkippedWork,
    build_mu2_index,
    index_file,
    parse_note_name,
    printed_key_signature,
    printed_rounding,
    read_mu2,
    symbtr_mu2,
    symbtr_pdf,
)

STAFF = re.compile(r"-\d+$")


def staffs_of(lines: list[str]) -> list[list[EncodedSymbol]]:
    """The token rows of one work, staff by staff, in reading order."""
    staffs = []
    for line in sorted(lines):
        path = line.strip().split(",")[1]
        if not os.path.exists(path):
            continue
        symbols = []
        for row in open(path, encoding="utf-8"):
            parts = row.split()
            if len(parts) == 5:
                symbols.append(EncodedSymbol(*parts))
        staffs.append(symbols)
    return staffs


def sounding_sequence(staffs: list[list[EncodedSymbol]]) -> list[tuple[str, int] | None]:
    """One entry per engraved event: its pitch and the commas the resolver gives it.

    A note too long for a single symbol is written as tied halves; they are one
    event, so only the first of a tied group is taken.
    """
    sequence: list[tuple[str, int] | None] = []
    for staff in staffs:
        for symbol in resolve_sounding(staff):
            if not symbol.rhythm.startswith("note"):
                continue
            if "tieStop" in symbol.articulation:
                continue
            commas = commas_for_lift(symbol.sounding) if symbol.sounding else 0
            sequence.append((symbol.pitch, commas if commas is not None else 0))
    return sequence


def wanted_sequence(stem: str, mu2_stem: str) -> tuple[list[tuple[str, int]], dict[int, int]]:
    """What the .mu2 says every note sounds, and the rounding the page applies.

    A few works file their .mu2 under a different name than their pdf, so the
    two are looked up separately.
    """
    import fitz  # noqa: PLC0415

    score = read_mu2(os.path.join(symbtr_mu2, mu2_stem + ".mu2"))
    with fitz.open(os.path.join(symbtr_pdf, stem + ".pdf")) as document:
        score.key = printed_key_signature(document, score.key)
        rounding = printed_rounding(document)
    wanted = []
    for event in score.events:
        parsed = parse_note_name(event["name"])
        if parsed is not None:
            wanted.append(parsed)
    return wanted, rounding


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Check only N works.")
    options = parser.parse_args()

    by_work: dict[str, list[str]] = collections.defaultdict(list)
    with open(index_file, encoding="utf-8") as handle:
        for line in handle:
            name = os.path.basename(line.strip().split(",")[0])[: -len(".png")]
            by_work[STAFF.sub("", name)].append(line)

    works = sorted(by_work)
    if options.limit:
        works = works[: options.limit]
    eprint(f"Checking {len(works)} works")

    index = build_mu2_index()
    notes = exact = rounded = wrong = 0
    length_mismatch = 0
    examples: list[str] = []
    causes: collections.Counter = collections.Counter()
    rounding_loss: collections.Counter = collections.Counter()

    for stem in works:
        try:
            wanted, rounding = wanted_sequence(stem, index.get(stem) or stem)
        except (SkippedWork, FileNotFoundError):
            continue
        got = sounding_sequence(staffs_of(by_work[stem]))
        if len(got) != len(wanted):
            length_mismatch += 1
            continue
        for (pitch, want), entry in zip(wanted, got):
            notes += 1
            if entry is None:
                wrong += 1
                continue
            _, have = entry
            if have == want:
                exact += 1
            elif rounding.get(want, want) == have:
                rounded += 1
                rounding_loss[f"{want:+d} reads back as {have:+d}"] += 1
            else:
                wrong += 1
                causes[f"{want:+d} came back {have:+d}"] += 1
                if len(examples) < 8:
                    examples.append(f"{stem[:44]} {pitch} wanted {want:+d} got {have:+d}")

    eprint(f"\nnotes checked            : {notes}")
    if notes:
        eprint(f"  came back exactly      : {exact} = {100 * exact / notes:.2f}%")
        eprint(f"  lost only the rounding : {rounded} = {100 * rounded / notes:.2f}%")
        eprint(f"  wrong                  : {wrong} = {100 * wrong / notes:.2f}%")
    eprint(f"works whose lengths disagree: {length_mismatch}")
    if rounding_loss:
        eprint("\nwhat the rounding costs:")
        for loss, count in rounding_loss.most_common():
            eprint(f"   {loss:<28} {count}")
    if causes:
        eprint("\nwhat goes wrong:")
        for cause, count in causes.most_common(10):
            eprint(f"   {cause:<28} {count}")
    for example in examples:
        eprint(f"   {example}")


if __name__ == "__main__":
    main()
