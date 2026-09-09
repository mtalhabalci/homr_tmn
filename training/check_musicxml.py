"""Does the MusicXML we write say what the score says?

The model's reading is measured elsewhere. This measures the stage after it: the
symbols become a MusicXML file, and the question is whether the notes in that
file carry the pitches and durations of the original .mu2. Perfect symbols go in
-- the labels themselves, not a model's guess -- so anything wrong here belongs
to the writing, not to the reading.

    python -m training.check_musicxml
    python -m training.check_musicxml --limit 100 --save demo.musicxml

Reports notes whose pitch or duration comes out wrong and works that fail to
write at all. A note the score engraves with a rounded sign -- the 2-comma flat
printed as the 1-comma one -- cannot come back exact and is counted apart: that
loss is in the page, and training/check_resolver.py measures it on its own.
"""

import argparse
import collections
import os
import re
import xml.etree.ElementTree as ElementTree
from fractions import Fraction

from homr.music_xml_generator import XmlGeneratorArguments, generate_xml
from homr.simple_logging import eprint
from homr.transformer.vocabulary import EncodedSymbol
from training.datasets.convert_symbtr import (
    SkippedWork,
    build_mu2_index,
    index_file,
    parse_note_name,
    printed_rounding,
    read_mu2,
    symbtr_mu2,
    symbtr_pdf,
)

STAFF = re.compile(r"-\d+$")
COMMAS_PER_SEMITONE = Fraction(9, 2)


def symbols_of(lines: list[str]) -> list[EncodedSymbol]:
    """Every token row of one work, staff by staff, in reading order."""
    symbols = []
    for line in sorted(lines):
        path = line.strip().split(",")[1]
        if not os.path.exists(path):
            continue
        for row in open(path, encoding="utf-8"):
            parts = row.split()
            if len(parts) == 5:
                symbols.append(EncodedSymbol(*parts))
    return symbols


def notes_in_xml(text: str) -> list[tuple[str, int, Fraction]]:
    """Pitch, commas and duration of every note, tied halves rejoined.

    A note too long for one symbol is engraved as tied halves and written as two
    <note> elements; the score has one note there, so the halves are added back
    together.
    """
    root = ElementTree.fromstring(text)
    divisions = 1
    found: list[tuple[str, int, Fraction]] = []
    for measure in root.iter("measure"):
        for element in measure.iter("divisions"):
            divisions = int(element.text or 1)
        for note in measure.iter("note"):
            pitch = note.find("pitch")
            if pitch is None or note.find("grace") is not None:
                continue
            step = pitch.findtext("step", "")
            octave = pitch.findtext("octave", "")
            alter = float(pitch.findtext("alter", "0"))
            commas = round(alter * float(COMMAS_PER_SEMITONE))
            # MusicXML counts in quarter notes, the .mu2 in whole ones.
            length = Fraction(int(note.findtext("duration", "0")), divisions * 4)
            ties = {tied.get("type") for tied in note.iter("tied")}
            if "stop" in ties and found:
                previous = found[-1]
                found[-1] = (previous[0], previous[1], previous[2] + length)
                continue
            found.append((f"{step}{octave}", commas, length))
    return found


def rounding_of(stem: str) -> dict[int, int]:
    """Which comma values this score engraves with a neighbour's sign."""
    import fitz  # noqa: PLC0415

    path = os.path.join(symbtr_pdf, stem + ".pdf")
    if not os.path.exists(path):
        return {}
    with fitz.open(path) as document:
        return printed_rounding(document)


def notes_in_mu2(stem: str) -> list[tuple[str, int, Fraction]]:
    score = read_mu2(os.path.join(symbtr_mu2, stem + ".mu2"))
    wanted = []
    for event in score.events:
        parsed = parse_note_name(event["name"])
        if parsed is None or event["grace"]:
            continue
        wanted.append((parsed[0], parsed[1], event["duration"]))
    return wanted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Check only N works.")
    parser.add_argument("--save", default=None, help="Write one work's xml here.")
    options = parser.parse_args()

    by_work: dict[str, list[str]] = collections.defaultdict(list)
    with open(index_file, encoding="utf-8") as handle:
        for line in handle:
            name = os.path.basename(line.strip().split(",")[0])[: -len(".png")]
            by_work[STAFF.sub("", name)].append(line)

    works = sorted(by_work)
    if options.limit:
        works = works[: options.limit]
    eprint(f"Writing MusicXML for {len(works)} works")

    index = build_mu2_index()
    checked = failed = counted_wrong = 0
    notes = right = wrong_pitch = wrong_length = rounded = 0
    faults: collections.Counter = collections.Counter()
    examples: list[str] = []

    for position, stem in enumerate(works):
        try:
            wanted = notes_in_mu2(index.get(stem) or stem)
        except (SkippedWork, FileNotFoundError):
            continue
        rounding = rounding_of(stem)
        symbols = symbols_of(by_work[stem])
        if not symbols:
            continue
        try:
            text = generate_xml(XmlGeneratorArguments(True), [symbols], stem).to_string()
        except Exception as error:  # noqa: BLE001
            failed += 1
            faults[type(error).__name__ + ": " + str(error)[:70]] += 1
            continue
        if options.save and position == 0:
            with open(options.save, "w", encoding="utf-8") as handle:
                handle.write(text)
            eprint(f"  wrote {options.save} for {stem[:50]}")

        got = notes_in_xml(text)
        checked += 1
        if len(got) != len(wanted):
            counted_wrong += 1
            if len(examples) < 6:
                examples.append(f"{stem[:44]}: {len(wanted)} notes, xml has {len(got)}")
            continue
        for (pitch, commas, length), (out_pitch, out_commas, out_length) in zip(wanted, got):
            notes += 1
            if pitch == out_pitch and rounding.get(commas, commas) == out_commas != commas:
                rounded += 1
            elif pitch != out_pitch or commas != out_commas:
                wrong_pitch += 1
                if len(examples) < 6:
                    examples.append(
                        f"{stem[:36]} {pitch}{commas:+d} came out {out_pitch}{out_commas:+d}"
                    )
            elif length != out_length:
                wrong_length += 1
                if len(examples) < 6:
                    examples.append(f"{stem[:36]} {pitch} {length} came out {out_length}")
            else:
                right += 1

    eprint(f"\nworks written        : {checked}")
    eprint(f"works that would not : {failed}")
    eprint(f"works with a different number of notes : {counted_wrong}")
    if notes:
        eprint(f"\nnotes compared       : {notes}")
        eprint(f"  pitch and length right : {right} = {100 * right / notes:.2f}%")
        eprint(f"  lost only the rounding : {rounded} = {100 * rounded / notes:.2f}%")
        eprint(f"  wrong pitch            : {wrong_pitch} = {100 * wrong_pitch / notes:.2f}%")
        eprint(f"  wrong length           : {wrong_length} = {100 * wrong_length / notes:.2f}%")
    if faults:
        eprint("\nwhy a work would not write:")
        for fault, count in faults.most_common(8):
            eprint(f"   {count:>4}  {fault}")
    if examples:
        eprint("\nexamples:")
        for example in examples:
            eprint(f"   {example}")


if __name__ == "__main__":
    main()
