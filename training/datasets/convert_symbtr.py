"""Convert the SymbTr corpus into homr training samples.

SymbTr (https://github.com/MTG/SymbTr) is a collection of ~2200 Turkish makam
scores, published in four parallel formats. Two of them are used here:

* the **PDF**, which is a real Mus2 engraving. 1996 of the 2200 embed a font
  named "Mus2" and draw the notation as text, so every notehead, clef, rest and
  accidental is a character with exact coordinates. That is the image the model
  has to read, and it is also how we locate the staves and the barlines.
* the **.mu2**, Mus2's own score format. It carries the makam key signature, the
  usul, the repeat structure, the grace notes and the pitch of every note in the
  printed (not repeat-expanded) order. That is where the labels come from.

The MusicXML that ships with SymbTr is deliberately *not* used. Its converter
dropped the 5-comma sharp entirely, collapsed the ten-glyph comma alphabet down
to four values, cannot express a three-accidental makam key signature, and
contains no grace notes, ties or repeats at all.

Alignment works through measures rather than individual notes: barlines are read
off the page, measures are cut out of the .mu2 by accumulating durations, and a
work is only accepted when every measure closes exactly on the usul length and
the two measure counts agree. Roughly 70% of the corpus passes; the rest is
skipped and reported.
"""

import argparse
import collections
import os
import random
import re
import shutil
import sys
from fractions import Fraction
from pathlib import Path

from homr.simple_logging import eprint
from homr.transformer.vocabulary import EncodedSymbol, aeu_commas, empty, key_accidental

script_location = os.path.dirname(os.path.realpath(__file__))
git_root = Path(script_location).parent.parent.absolute()
dataset_root = os.path.join(git_root, "datasets")
symbtr_root = os.path.join(dataset_root, "SymbTr-2.0.0")
symbtr_pdf = os.path.join(symbtr_root, "pdf")
symbtr_mu2 = os.path.join(symbtr_root, "mu2")
working_dir = os.path.join(dataset_root, "SymbTr-work")

index_file = os.path.join(symbtr_root, "index.txt")
index_train = os.path.join(symbtr_root, "index_train.txt")
index_val = os.path.join(symbtr_root, "index_val.txt")
index_test = os.path.join(symbtr_root, "index_test.txt")
# Named the way the other converters name theirs, so train.py can pick it up.
symbtr_train_index = index_train
symbtr_val_index = index_val
symbtr_test_index = index_test

# Staff images are rendered at the same scale convert_lieder uses, so that a
# staff coming from either dataset lands on the model canvas the same way.
target_page_width = 1400

# --- .mu2 record codes ------------------------------------------------------
# Only these advance musical time. Codes 50-63 are header rows and carry a
# duration that is *not* a note - reading them as notes was what originally made
# the measures overflow. Code 8 is a çarpma (grace note): printed, but it takes
# no time of its own.
NOTE_CODES = frozenset({"9", "1", "7", "10", "11", "12", "23", "24"})
GRACE_CODES = frozenset({"8"})
KEY_SIGNATURE_CODE = "50"

# Section and repeat marks. They sit in the two lyric columns of a marker row.
OPENING_MARKS = {
    "(": "repeatStart",
    "(1": "voltaStart",
    "(2": "voltaStart",
    "[": "repeatStart",
    "{": "repeatStart",
}
CLOSING_MARKS = {
    ":": "repeatEnd",
    ")": "voltaStop",
    "]": "repeatEnd",
    "$": "bolddoublebarline",
    "^": "repeatEnd",
    "}": "voltaStop",
}

TURKISH_NOTE_NAMES = {
    "do": "C",
    "re": "D",
    "mi": "E",
    "fa": "F",
    "sol": "G",
    "la": "A",
    "si": "B",
}

# The koma glyphs that are scarce enough to need oversampling in training. The
# 5-comma sharp (eviç) is the one this dataset exists for and appears on the
# order of a hundred times, against tens of thousands for the common glyphs.
RARE_LIFTS = ("sharp5", "sharp2", "sharp8", "flat8")
# A class the model never sees enough of stays at zero recall, so scarce staffs
# get repeated. Repeating by a flat factor overshot badly: sharp5 ended up
# eleven times more common in training than on a real page, and the model
# learned to reach for it whenever it was unsure -- 58% recall at 8% precision,
# eating flat1 and flat2 along the way. Count up to a target instead, so every
# class becomes learnable without any class acquiring a false prior. flat3 sits
# near this figure with 1133 examples and reaches 81% recall, which is what the
# target is calibrated on.
RARE_LIFT_TARGET = 1500
# Where a sign sits matters more than how often it appears. Every class reads at
# essentially 100% in the key signature, where it stands at the head of the staff
# in a place the model can expect. Among the notes the same classes read at
# 92-97% -- except the five-comma sharp, at 39%, which the corpus trains there
# only 69 times against 689 in a signature. Recall among the notes tracks that
# count: above 300 examples every class clears 92%, and the two that fall below
# it are the two that fail.
#
# Height, the first suspect, does not hold up: the sign moved to a degree the
# corpus never puts it on still read perfectly, as long as it stayed in the
# signature.
RARE_CONTEXT_TARGET = 400
MAX_REPEATS = 8
VAL_FRACTION = 0.1
TEST_FRACTION = 0.1
# A glyph carried by fewer works than this cannot be split three ways, so every
# work holding it goes to training. Above the threshold the split stays random,
# which matters: if all the eviç works sat in training we could never measure
# how well eviç is recognised.
MIN_WORKS_TO_SPLIT = 4


class SkippedWork(Exception):
    """A work that cannot be aligned confidently enough to train on."""


def build_mu2_index() -> dict[str, str]:
    """Map each PDF stem to its .mu2 stem.

    Most of the corpus uses the same filename on both sides. About 500 works do
    not: the aranagme and seyir pieces carry a placeholder title in one folder
    and not the other ("...--agiraksak--1--" against "...--agiraksak----001"),
    and a few abbreviate the composer. Where makam, form and usul identify
    exactly one unmatched file on each side, pair them up. A wrong pairing is
    not dangerous here - the measure check in convert_work rejects it, because
    another work's durations will not close on this work's usul.
    """
    pdf_stems = {path.stem for path in Path(symbtr_pdf).glob("*.pdf")}
    mu2_stems = {path.stem for path in Path(symbtr_mu2).glob("*.mu2")}
    index = {stem: stem for stem in pdf_stems & mu2_stems}

    def group(stems: set[str]) -> dict[tuple[str, ...], list[str]]:
        grouped: dict[tuple[str, ...], list[str]] = {}
        for stem in stems:
            grouped.setdefault(tuple(stem.split("--")[:3]), []).append(stem)
        return grouped

    unmatched_pdf = group(pdf_stems - mu2_stems)
    unmatched_mu2 = group(mu2_stems - pdf_stems)
    for key, candidates in unmatched_pdf.items():
        others = unmatched_mu2.get(key, [])
        if len(candidates) == 1 and len(others) == 1:
            index[candidates[0]] = others[0]
    return index


# --- reading the .mu2 -------------------------------------------------------


def parse_note_name(name: str) -> tuple[str, int] | None:
    """'La4b5' or 'A4b5' -> ('A4', -5). Returns None for a rest."""
    if not name or name.upper() == "ES":
        return None
    text = name.strip()
    lowered = text.lower()
    root = next(
        (r for r in ("sol", "do", "re", "mi", "fa", "la", "si") if lowered.startswith(r)),
        None,
    )
    if root is not None:
        letter, remainder = TURKISH_NOTE_NAMES[root], text[len(root) :]
    elif text[0].upper() in "ABCDEFG":
        # The key signature is written with letters, the notes with solfège.
        letter, remainder = text[0].upper(), text[1:]
    else:
        return None
    if not remainder or not remainder[0].isdigit():
        return None
    octave, accidental = remainder[0], remainder[1:]
    commas = 0
    if accidental:
        sign = 1 if accidental[0] == "#" else (-1 if accidental[0] == "b" else 0)
        if sign and accidental[1:].isdigit():
            commas = sign * int(accidental[1:])
    return letter + octave, commas


def lift_for_commas(commas: int) -> str:
    if commas == 0:
        return "N"
    if abs(commas) not in aeu_commas:
        # A handful of works use 6- and 7-comma steps that the vocabulary has no
        # token for. Skip the work rather than mislabel the note as its neighbour.
        raise SkippedWork(f"{abs(commas)}-comma accidental is outside the vocabulary")
    return f"{'sharp' if commas > 0 else 'flat'}{abs(commas)}"


class Mu2Score:
    def __init__(self, numerator: int, denominator: int, key: str, events: list[dict]):
        self.numerator = numerator
        self.denominator = denominator
        self.key = key
        self.events = events

    @property
    def measure_length(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)


def read_mu2(path: str) -> Mu2Score:
    lines = Path(path).read_text(encoding="cp1254", errors="replace").splitlines()
    if not lines:
        raise SkippedWork("empty .mu2")
    header = lines[0].split("\t")
    if len(header) < 2 or not (header[0].strip().isdigit() and header[1].strip().isdigit()):
        raise SkippedWork("no time signature in the .mu2 header")
    key = ""
    events: list[dict] = []
    pending_open: str | None = None
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) < 9:
            continue
        code = fields[0].strip()
        if code == KEY_SIGNATURE_CODE:
            key = fields[8].strip()
            continue
        opening, closing = fields[7].strip(), fields[8].strip()
        marks: tuple[str | None, str | None] | None = None
        if opening in OPENING_MARKS or closing in CLOSING_MARKS:
            marks = (OPENING_MARKS.get(opening), CLOSING_MARKS.get(closing))

        numerator, denominator = fields[2].strip(), fields[3].strip()
        sounds = (
            code in NOTE_CODES
            and numerator.isdigit()
            and denominator.isdigit()
            and int(numerator)
            and int(denominator)
        )
        # What makes a row a marker is that it carries a mark and no note --
        # not its code, because a bare repeat sign is often written with a note
        # code and every note field left empty. Going by the code alone loses
        # those signs entirely; going by the mark alone threw away the 18,265
        # notes that share their row with one, across 1,617 of the 2,200 works,
        # which taught the model to skip a note plainly on the page and left the
        # measures short so the work was dropped for not closing.
        if marks and not sounds and code not in GRACE_CODES:
            open_mark, close_mark = marks
            if close_mark and events:
                # A closing sign ends the section it follows, so it belongs to
                # the note before it. Carrying it forward like an opening put it
                # one measure late, and a second marker row right after -- the
                # usual "end this repeat, start the next" pair -- overwrote it
                # outright. Between them the model saw repeatEnd on the wrong
                # barline more often than the right one and never learned it.
                events[-1]["close"] = close_mark
            if open_mark:
                pending_open = open_mark
            continue
        if code in GRACE_CODES:
            grace = {"name": fields[1].strip(), "duration": Fraction(0), "grace": True}
            if marks:
                grace["open"], grace["close"] = marks
            events.append(grace)
            continue

        if not sounds:
            continue
        event = {
            "name": fields[1].strip(),
            "duration": Fraction(int(numerator), int(denominator)),
            "grace": False,
        }
        if pending_open or marks:
            here = marks or (None, None)
            event["open"] = pending_open or here[0]
            event["close"] = here[1]
            pending_open = None
        events.append(event)
    if not events:
        raise SkippedWork("no notes in the .mu2")
    return Mu2Score(int(header[0]), int(header[1]), key, events)


# --- durations --------------------------------------------------------------


def _duration_units(bases: tuple[int, ...]) -> list[tuple[Fraction, int, int]]:
    units = [
        (Fraction(1, base) * (Fraction(2) - Fraction(1, 2**dots)), base, dots)
        for base in bases
        for dots in (0, 1, 2)
    ]
    units.sort(key=lambda unit: -unit[0])
    return units


_PLAIN_UNITS = _duration_units((1, 2, 4, 8, 16, 32, 64))
_TUPLET_UNITS = _duration_units((1, 2, 4, 8, 16, 32, 64, 3, 6, 12, 24, 48))


def _single_duration(duration: Fraction) -> tuple[int, int] | None:
    for dots in (0, 1, 2):
        base = duration / (Fraction(2) - Fraction(1, 2**dots))
        if base.numerator == 1:
            return base.denominator, dots
    return None


def _greedy_split(
    duration: Fraction, units: list[tuple[Fraction, int, int]]
) -> list[tuple[int, int]] | None:
    parts: list[tuple[int, int]] = []
    remainder = duration
    for value, base, dots in units:
        while remainder >= value and len(parts) < 4:
            parts.append((base, dots))
            remainder -= value
        if remainder == 0:
            return parts
    return None


def split_duration(duration: Fraction) -> list[tuple[int, int]]:
    """Durations a single notehead cannot express become tied notes.

    Plain (power of two) values are tried first: 5/8 should come out as a half
    tied to an eighth, not as a pair of odd tuplet fragments that happen to add
    up to the same length.
    """
    single = _single_duration(duration)
    if single:
        return [single]
    return _greedy_split(duration, _PLAIN_UNITS) or _greedy_split(duration, _TUPLET_UNITS) or []


class EngravingState:
    """Works out which accidentals are actually printed in front of the notes.

    The .mu2 spells out the sounding pitch of every note, but a printed score
    shows an accidental only where the note departs from what the reader already
    knows: the key signature at the head of the staff, plus any accidental set
    earlier in the same measure. Roughly four out of five altered notes carry no
    sign at all.

    The label has to be what is on the page, because the page is all the model
    ever sees. Recovering the sounding pitch from the printed symbols is the
    reader's job, and downstream ours - see homr/circle_of_fifths.py.
    """

    def __init__(self, key: str, rounding: dict[int, int] | None = None):
        self.signature: dict[str, int] = {}
        for entry in (part.strip() for part in key.split("/")):
            parsed = parse_note_name(entry) if entry else None
            if parsed:
                self.signature[parsed[0][0]] = parsed[1]
        self.measure: dict[str, int] = {}
        self.rounding = rounding or {}

    def start_measure(self) -> None:
        self.measure = {}

    def drawn_lift(self, pitch: str, commas: int) -> str:
        """The accidental printed before this note, or empty if there is none."""
        letter, expected = pitch[0], self.measure.get(pitch)
        if expected is None:
            expected = self.signature.get(letter, 0)
        # Compare what would be *printed*, not what sounds. AEU has no sign for a
        # 2- or 3-comma step and most scores print the nearest one it does have
        # (see printed_rounding), which makes two comma values share a glyph.
        # Moving between them changes nothing on the page, so nothing is drawn.
        shown = self.rounding.get(commas, commas)
        if shown == self.rounding.get(expected, expected):
            return empty
        self.measure[pitch] = commas
        return lift_for_commas(shown)


# Mus2 never prints a dotted rest. A dotted quarter's worth of silence is a
# quarter rest followed by an eighth rest -- 36 of 37 times on the page, the
# odd one out reversed -- so a rest is split into plain values, largest first.
_UNDOTTED_UNITS = [unit for unit in _PLAIN_UNITS if unit[2] == 0]


def rest_parts(duration: Fraction) -> list[tuple[int, int]]:
    return _greedy_split(duration, _UNDOTTED_UNITS) or split_duration(duration)


def tokens_for_event(
    event: dict, unresolved: collections.Counter, state: EngravingState | None = None
) -> list[EncodedSymbol]:
    pitch = parse_note_name(event["name"])
    if event["grace"]:
        parts, suffix = [(8, 0)], "G"
    else:
        split = split_duration if pitch is not None else rest_parts
        parts, suffix = split(event["duration"]), ""
        if not parts:
            unresolved[str(event["duration"])] += 1
            return []
    # The accidental is whatever the engraver put on the page. Tied halves of one
    # note share a notehead's worth of ink, so only the first can carry a sign.
    if pitch is None:
        lift = "_"
    elif state is None:
        lift = lift_for_commas(pitch[1])
    else:
        lift = state.drawn_lift(pitch[0], pitch[1])

    symbols = []
    for index, (base, dots) in enumerate(parts):
        kern = f"{base}{'.' * dots}{suffix}"
        if pitch is None:
            symbols.append(EncodedSymbol(f"rest_{kern}", "_", "_", "_", "upper"))
            continue
        if len(parts) == 1:
            articulation = "_"
        elif index == 0:
            articulation = "tieStart"
        else:
            articulation = "tieStop"
        symbols.append(
            EncodedSymbol(
                f"note_{kern}", pitch[0], lift if index == 0 else "_", articulation, "upper"
            )
        )
    return symbols


# --- reading the page -------------------------------------------------------


def _staff_lines(page: "fitz.Page") -> list[list[float]]:  # noqa: F821
    """Group the long horizontal rules of a page into staves of five."""
    width = page.rect.width
    positions = []
    for drawing in page.get_drawings():
        for item in drawing["items"]:
            if item[0] == "l":
                start, end = item[1], item[2]
                if abs(start.y - end.y) < 0.6 and abs(end.x - start.x) > width * 0.5:
                    positions.append(round((start.y + end.y) / 2, 1))
            elif item[0] == "re":
                rect = item[1]
                if rect.height < 1.2 and rect.width > width * 0.5:
                    positions.append(round(rect.y0, 1))
    positions = sorted(set(positions))
    if not positions:
        return []
    staves, current = [], [positions[0]]
    for y in positions[1:]:
        if y - current[-1] < 14:
            current.append(y)
        else:
            staves.append(current)
            current = [y]
    staves.append(current)
    return [staff for staff in staves if len(staff) >= 4]


# Glyphs that are, or contain, a notehead. A vertical line touching one of these
# is a stem, not a barline - without this test the barline count roughly doubles.
NOTEHEAD_GLYPHS = frozenset(
    {0x78, 0x2A, 0x2B, 0x6F, 0x25, 0x27, 0x28, 0x29, 0x2C, 0x74}
)

# The Mus2 font draws one accidental per comma value. Reading them back tells us
# which signs a particular score actually uses.
ACCIDENTAL_GLYPHS = {
    0x54: -1, 0x53: -2, 0x52: -3, 0x51: -4, 0x50: -5,
    0x55: 1, 0x56: 2, 0x57: 3, 0x58: 4, 0x59: 5, 0x5C: 8,
}
# Where AEU has no sign, the nearest one it does have.
NEAREST_AEU = {2: 1, 3: 4, -2: -1, -3: -4}


def _key_signature_glyphs(document: "fitz.Document") -> list[int] | None:  # noqa: F821
    """Character codes of the key signature on the first staff, left to right.

    Returns None when the codes cannot be trusted. The Mus2 font is subsetted
    per PDF, and while nearly every file uses the same encoding, a few do not:
    in gerdaniye--turku--aksak--salina_salina the 2-comma flat comes through as
    0x71, which elsewhere is the treble clef. Reading that file's glyphs would
    say the key signature is smaller than it is and mark every note it covers.
    """
    page = document[0]
    staves = _staff_lines(page)
    if not staves:
        return None
    top, bottom = staves[0][0], staves[0][-1]
    height = bottom - top
    found = sorted(
        (glyph[0], glyph[4])
        for glyph in _mus2_glyphs(page)
        if glyph[5] >= 24 and glyph[0] < 60
        and top - height < (glyph[2] + glyph[3]) / 2 < bottom + height
    )
    codes = [code for _, code in found]
    if any(code not in ACCIDENTAL_GLYPHS for code in codes):
        return None
    return codes


def printed_key_signature(document: "fitz.Document", key: str) -> str:  # noqa: F821
    """Trim the .mu2 key signature down to the accidentals actually engraved.

    The two do not always agree. A score may leave an accidental out of the
    printed signature and mark that note individually instead - the .mu2 for
    hicaz--sarki--turkaksagi--solsan_da_sararsan lists three, the page shows two,
    and the missing sharp appears on twenty-five noteheads. Believing the .mu2
    there would suppress every one of those signs in the labels.

    The printed glyphs are matched to the .mu2 entries in order by comma value,
    which is enough: a signature never repeats a letter.
    """
    tokens = [part.strip() for part in key.split("/") if part.strip()]
    codes = _key_signature_glyphs(document)
    if not tokens or codes is None:
        return key
    remaining = [ACCIDENTAL_GLYPHS[code] for code in codes]
    kept = []
    for token in tokens:
        parsed = parse_note_name(token)
        if not parsed:
            continue
        # The signature itself may be engraved with the rounded sign, so a
        # 2-comma entry can show up on the page as the 1-comma glyph.
        for candidate in (parsed[1], NEAREST_AEU.get(parsed[1])):
            if candidate is not None and candidate in remaining:
                remaining.remove(candidate)
                kept.append(token)
                break
    return "/".join(kept)


def printed_rounding(document: "fitz.Document") -> dict[int, int]:  # noqa: F821
    """Decide whether this score rounds its 2- and 3-comma accidentals.

    Those two steps sit outside AEU, and a score either prints the sign Mus2 has
    for them or rounds to the nearest AEU sign - consistently, one way per file.
    If a value's own sign appears nowhere on the page, the score rounds it.
    """
    if _key_signature_glyphs(document) is None:
        # This file does not use the usual encoding, so an absent code proves
        # nothing. Take the .mu2 at face value rather than guess.
        return {}
    seen: set[int] = set()
    for page in document:
        for glyph in _mus2_glyphs(page):
            commas = ACCIDENTAL_GLYPHS.get(glyph[4])
            if commas is not None:
                seen.add(commas)
    return {value: nearest for value, nearest in NEAREST_AEU.items() if value not in seen}


def _mus2_glyphs(page: "fitz.Page") -> list[tuple]:  # noqa: F821
    glyphs = []
    for block in page.get_text("rawdict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                if "Mus2" not in span.get("font", ""):
                    continue
                for char in span.get("chars", []):
                    x0, y0, x1, y1 = char["bbox"]
                    glyphs.append((x0, x1, y0, y1, ord(char["c"]), round(span["size"], 1)))
    return glyphs


def _barlines(page: "fitz.Page", staff: list[float], glyphs: list[tuple]) -> list[float]:  # noqa: F821, E501
    top, bottom = staff[0], staff[-1]
    height = bottom - top
    candidates = []
    for drawing in page.get_drawings():
        for item in drawing["items"]:
            if item[0] == "l":
                start, end = item[1], item[2]
                if abs(start.x - end.x) < 0.9:
                    y0, y1 = sorted((start.y, end.y))
                    candidates.append(((start.x + end.x) / 2, y0, y1))
            elif item[0] == "re":
                rect = item[1]
                if rect.width < 4.0 and rect.height > height * 0.5:
                    candidates.append(((rect.x0 + rect.x1) / 2, rect.y0, rect.y1))
    # A barline spans the staff exactly; a stem overshoots or stops short.
    spanning = [
        x
        for x, y0, y1 in candidates
        if abs(y0 - top) < 2.5 and abs(y1 - bottom) < 2.5
    ]
    heads = [
        glyph
        for glyph in glyphs
        if glyph[4] in NOTEHEAD_GLYPHS
        and glyph[5] >= 20
        and top - height < (glyph[2] + glyph[3]) / 2 < bottom + height
    ]
    without_stems = [
        x for x in spanning if not any(head[0] - 2.5 <= x <= head[1] + 2.5 for head in heads)
    ]
    # A repeat barline is drawn as a thick rule plus a thin one; count it once.
    without_stems.sort()
    merged: list[float] = []
    for x in without_stems:
        if merged and x - merged[-1] < 5:
            continue
        merged.append(x)
    # Anything before the first notehead opens a measure rather than closing one.
    first_head = min((head[0] for head in heads), default=None)
    if first_head is not None:
        merged = [x for x in merged if x > first_head]
    return merged


# --- putting the two together ----------------------------------------------


def _cut_into_measures(score: Mu2Score) -> list[list[dict]]:
    measures: list[list[dict]] = []
    current: list[dict] = []
    accumulated = Fraction(0)
    exact = 0
    for event in score.events:
        current.append(event)
        accumulated += event["duration"]
        if accumulated >= score.measure_length:
            if accumulated == score.measure_length:
                exact += 1
            measures.append(current)
            current, accumulated = [], accumulated - score.measure_length
    if current:
        measures.append(current)
    if exact != len(measures):
        raise SkippedWork(
            f"{len(measures) - exact} of {len(measures)} measures do not close on the usul"
        )
    return measures


def _header(score: Mu2Score, state: EngravingState, is_first_staff: bool) -> list[EncodedSymbol]:
    # Mus2 redraws the clef and the whole key signature at the head of every
    # system, so every staff sample carries them; only the time signature is
    # printed once, at the start of the piece. The staff image is all the model
    # ever sees, so the labels have to match what is on that one staff.
    tokens: list[EncodedSymbol] = [EncodedSymbol("clef_G2", "_", "_", "_", "upper")]
    for entry in (part.strip() for part in score.key.split("/")):
        pitch = parse_note_name(entry) if entry else None
        if pitch:
            # The signature is engraved under the same rounding as everything
            # else, so a 2-comma entry is drawn with the 1-comma sign. Labelling
            # it 2 asked the model to tell two works apart by a glyph they
            # share, once per staff, for the whole piece.
            shown = state.rounding.get(pitch[1], pitch[1])
            tokens.append(
                EncodedSymbol(key_accidental, pitch[0], lift_for_commas(shown), "_", "upper")
            )
    if is_first_staff:
        tokens.append(EncodedSymbol(f"timeSignature_{score.numerator}/{score.denominator}"))
    return tokens


def _measure_tokens(
    measure: list[dict], unresolved: collections.Counter, state: EngravingState
) -> list[EncodedSymbol]:
    opening = next((e["open"] for e in measure if e.get("open")), None)
    closing = next((e["close"] for e in measure if e.get("close")), None)
    tokens = [EncodedSymbol(opening)] if opening else []
    state.start_measure()
    for event in measure:
        tokens.extend(tokens_for_event(event, unresolved, state))
    tokens.append(EncodedSymbol(closing or "barline"))
    return tokens


# What reading the page changed, tallied over a run and reported at the end.
page_report: collections.Counter = collections.Counter()


def _is_head(token: EncodedSymbol) -> bool:
    """A note printed at full size; grace notes are small."""
    return token.rhythm.startswith("note") and not token.rhythm.endswith("G")


def _staff_cuts(
    body: list[EncodedSymbol],
    ends: list[int],
    staves: list[dict],
    counts: list[int],
    fits: "collections.abc.Callable[[int, int, int], bool]",
) -> list[tuple[int, int]]:
    """Where each staff's labels start and end in the work's token stream.

    Counting barlines is the obvious guide and usually right, but one missed or
    extra barline moves every later staff by a measure: the staff then carries
    labels for notes it does not show, and its neighbour for notes it does. And
    in the long usuls a measure does not always fit on one line; Mus2 breaks it
    and carries the rest over to the next staff.

    So each staff is given exactly as many notes as the page prints on it,
    trying in turn: a cut at a barline (the one nearest the barline count, when
    a measure of rests lets several qualify), then a cut inside a measure, and
    only then the barline count alone. A cut is taken only if the notes it
    gives the staff match the page pitch for pitch, so one unreadable staff
    costs itself and not the staffs after it.
    """
    heads_before = [0]
    for token in body:
        heads_before.append(heads_before[-1] + _is_head(token))
    cuts, start = [], 0
    for index, staff in enumerate(staves):
        first = next((m for m, end in enumerate(ends) if end > start), len(ends) - 1)
        guide = ends[min(first + max(len(staff["bars"]) - 1, 0), len(ends) - 1)]
        if index == len(staves) - 1:
            candidates = [len(body)]
        else:
            wanted = heads_before[start] + counts[index]
            at_barline = sorted(
                (end for end in ends if end > start and heads_before[end] == wanted),
                key=lambda end: (abs(end - guide), end),
            )
            inside = [
                k
                for k in range(start + 1, len(body) + 1)
                if heads_before[k] == wanted and _is_head(body[k - 1])
            ][:1]
            candidates = [*at_barline, *inside, guide]
        end = next((c for c in candidates if fits(index, start, c)), candidates[-1])
        if end not in ends:
            page_report["staffs ending inside a measure"] += 1
        elif end != guide:
            page_report["staffs given other measures than the barline count says"] += 1
        cuts.append((start, end))
        start = end
    return cuts


def _value(rhythm: str) -> Fraction:
    kern = rhythm.split("_", 1)[1].rstrip("G")
    dots = kern.count(".")
    return Fraction(1, int(kern.rstrip("."))) * (2 - Fraction(1, 2**dots))


def _kind(base: int) -> str:
    return {1: "1", 2: "2", 4: "4"}.get(base, "short")


def _ways(total: Fraction, parts: int) -> list[list[tuple[int, int]]]:
    """Every way to write a duration as so many tied plain values, in order."""
    if parts == 1:
        return [[(base, dots)] for value, base, dots in _PLAIN_UNITS if value == total]
    return [
        [(base, dots), *rest]
        for value, base, dots in _PLAIN_UNITS
        if value < total
        for rest in _ways(total - value, parts - 1)
    ]


def _splits_from_page(
    tokens: list[EncodedSymbol], where: list[int | None], heads: list[dict]
) -> list[EncodedSymbol]:
    """Split each long note the way the page splits it.

    A note too long for one notehead is written as tied notes, and there are
    usually several ways to do it: five eighths as a half and an eighth, or a
    dotted quarter and a quarter. split_duration takes the largest value first;
    Mus2 follows the beats of the usul, and of 428 tied pairs the two agreed
    once. The page tells which way it went: each head is a whole, a half, a
    quarter or shorter, and is dotted or not, and in 424 of the 428 exactly one
    way fits that. Where none or several fit, the label is left as it was.
    """
    result = list(tokens)
    notes = [i for i, t in enumerate(tokens) if t.rhythm.startswith("note")]
    k = 0
    while k < len(notes):
        first = tokens[notes[k]]
        group = [k]
        k += 1
        if first.articulation != "tieStart":
            continue
        while k < len(notes) and tokens[notes[k]].articulation == "tieStop" and tokens[notes[k]].pitch == first.pitch:
            group.append(k)
            k += 1
        if len(group) < 2 or any(where[g] is None for g in group):
            continue
        page = [(heads[where[g]]["kind"], int(heads[where[g]]["dotted"])) for g in group]
        if any(kind is None for kind, _ in page):
            continue
        labelled = []
        for g in group:
            kern = tokens[notes[g]].rhythm.split("_", 1)[1]
            labelled.append((_kind(int(kern.rstrip("."))), kern.count(".")))
        if labelled == page:
            continue
        total = sum(_value(tokens[notes[g]].rhythm) for g in group)
        fitting = [way for way in _ways(total, len(group)) if [(_kind(b), d) for b, d in way] == page]
        if len(fitting) != 1:
            page_report["tied notes the page does not settle"] += 1
            continue
        for g, (base, dots) in zip(group, fitting[0]):
            old = tokens[notes[g]]
            result[notes[g]] = EncodedSymbol(
                f"note_{base}{'.' * dots}", old.pitch, old.lift, old.articulation, old.position
            )
        page_report["tied notes split the way the page splits them"] += 1
    return result


def _signature_from_page(
    tokens: list[EncodedSymbol], printed: list[tuple[str, str]]
) -> list[EncodedSymbol]:
    """Label the key signature with the signs printed at the head of the staff.

    printed_key_signature decides which .mu2 entries were engraved by comma
    value alone, and a hicaz signature of B-flat, F-sharp and C-sharp printed
    as B-flat and C-sharp comes out as B-flat and F-sharp: both sharps are
    four commas. The page says which note each sign sits on.
    """
    labelled = [(t.pitch, t.lift) for t in tokens if t.rhythm == key_accidental]
    if labelled == printed:
        return tokens
    page_report["staffs whose key signature now follows the page"] += 1
    rest = [t for t in tokens if t.rhythm != key_accidental]
    signature = [EncodedSymbol(key_accidental, pitch, lift, "_", "upper") for pitch, lift in printed]
    # The clef comes first, then the signature, then the rest of the staff.
    return rest[:1] + signature + rest[1:]


_CLOSINGS = ("barline", "repeatEnd", "bolddoublebarline", "voltaStop")
_OPENINGS = ("repeatStart", "voltaStart")
_STRUCTURE = _CLOSINGS + _OPENINGS


def _bars_from_page(  # noqa: PLR0913
    tokens: list[EncodedSymbol],
    where: list[int | None],
    heads: list[dict],
    marks: list[tuple[float, str]],
    hooks: list[tuple[float, str]],
    page_width: float,
) -> list[EncodedSymbol]:
    """Label each barline the way the page draws it.

    The .mu2 marks sections for Mus2 with a handful of characters, and the
    converter read every one of them as a repeat or a closing double bar. The
    page does not: a mark that ends a section may be printed as a plain
    barline with "[KARAR'a]" beneath it, a mark that opens one as a segno above
    the first measure, and the "$" that became bolddoublebarline is a segno
    too. Two labels in five for the repeat end stood over a plain barline.

    Between each pair of notes the page shows plain rules, or a repeat whose
    dots say which way it faces, or nothing; and above them a volta bracket
    may open or close. The labelled barline and volta marks between the same
    two notes are set to match: volta closings had drifted a measure early.
    """
    notes = [i for i, t in enumerate(tokens) if t.rhythm.startswith("note")]
    anchors = [(-1, 0.0)]
    for n, i in enumerate(notes):
        if where[n] is not None:
            anchors.append((i, heads[where[n]]["origin"][0]))
    anchors.append((len(tokens), page_width))

    result: list[EncodedSymbol] = []
    for (start, left), (end, right) in zip(anchors, anchors[1:]):
        if start >= 0:
            result.append(tokens[start])
        window = tokens[start + 1 : end]
        low, high = (left + 2 if start >= 0 else 0), right - 2
        shape = bar_shape_between(marks, low, high)
        opens = any(low < x < high and kind == "start" for x, kind in hooks)
        closes = any(low < x < high and kind == "end" for x, kind in hooks)
        result.extend(_rebar(window, shape, opens, closes, at_staff_start=start < 0))
    return result


def bar_shape_between(marks: list[tuple[float, str]], low: float, high: float) -> str:
    from training.datasets.page_notes import bar_shape  # noqa: PLC0415

    return bar_shape([kind for x, kind in marks if low < x < high])


def _rebar(
    window: list[EncodedSymbol], shape: str, opens: bool, closes: bool, at_staff_start: bool
) -> list[EncodedSymbol]:
    """The tokens between two notes, their barline and volta marks set from the page.

    Only a gap the labels already put a barline or volta mark in is touched: a
    mark the page shows where the labels see no measure boundary at all is a
    disagreement about the measures, which this cannot settle.
    """
    labelled = [t.rhythm for t in window if t.rhythm in _STRUCTURE]
    if not labelled or shape == "thick":
        return window
    if shape == "none" and not at_staff_start and not (opens or closes):
        return window
    # A bracket that closes on a plain barline takes the barline's place, as
    # the corpus has always written it; on a repeat it follows the repeat.
    stop = ["voltaStop"] if closes else []
    closing = {
        "none": stop,
        "plain": stop or ["barline"],
        "end": ["repeatEnd", *stop],
        "start": stop or ["barline"],
        "end+start": ["repeatEnd", *stop],
        "final": ["bolddoublebarline", *stop],
    }[shape]
    if at_staff_start:
        # Nothing ends before the first note of a line; only a repeat can open.
        closing = []
    opening = (["repeatStart"] if shape in ("start", "end+start") else []) + (["voltaStart"] if opens else [])
    wanted = closing + opening
    if wanted == labelled:
        return window
    page_report[f"barline {'+'.join(labelled)} -> {'+'.join(wanted) or 'nothing'}"] += 1
    first = next(i for i, t in enumerate(window) if t.rhythm in _STRUCTURE)
    kept = [t for t in window if t.rhythm not in _STRUCTURE]
    placed = first - sum(1 for t in window[:first] if t.rhythm in _STRUCTURE)
    return kept[:placed] + [EncodedSymbol(rhythm) for rhythm in wanted] + kept[placed:]


def _time_from_page(tokens: list[EncodedSymbol], printed: str | None) -> list[EncodedSymbol]:
    """Label the time signature as printed: a düyek can be engraved 8/8 where
    the .mu2 header says 4/4. A printed signature the vocabulary cannot spell
    is left as the .mu2 has it."""
    if printed is None or printed not in _rhythm_vocabulary():
        return tokens
    result = []
    for token in tokens:
        if token.rhythm.startswith("timeSignature") and token.rhythm != printed:
            page_report[f"time signature {token.rhythm[14:]} -> {printed[14:]}"] += 1
            token = EncodedSymbol(printed)
        result.append(token)
    return result


def _rhythm_vocabulary() -> dict:
    from homr.transformer.vocabulary import Vocabulary  # noqa: PLC0415

    global _RHYTHMS  # noqa: PLW0603
    if _RHYTHMS is None:
        _RHYTHMS = Vocabulary().rhythm
    return _RHYTHMS


_RHYTHMS: dict | None = None


def _signs_from_page(
    tokens: list[EncodedSymbol], where: list[int | None], heads: list[dict]
) -> list[EncodedSymbol]:
    """Label each note with the sign the page draws before it."""
    result, number = [], 0
    for token in tokens:
        if token.rhythm.startswith("note"):
            head = where[number]
            number += 1
            if head is not None:
                drawn = heads[head]["lift"] or empty
                if drawn != token.lift:
                    page_report[f"sign {token.lift} -> {drawn}"] += 1
                    token = token.change_lift(drawn)
        result.append(token)
    return result


def _index_line(base: str) -> str:
    """One 'image,tokens' index entry, always with forward slashes.

    The dataset is usually built on one machine and trained on another, so the
    paths must not carry a Windows separator into a Linux training run.
    """
    image = os.path.relpath(base + ".png", git_root).replace(os.sep, "/")
    tokens = os.path.relpath(base + ".tokens", git_root).replace(os.sep, "/")
    return f"{image},{tokens}\n"


def reindex() -> list[str]:
    """Rebuild the index from the staff files already in the working directory."""
    bases = sorted(str(path)[: -len(".tokens")] for path in Path(working_dir).glob("*.tokens"))
    return [_index_line(base) for base in bases]


def convert_work(
    stem: str,
    just_token_files: bool,
    unresolved: collections.Counter,
    mu2_stem: str | None = None,
) -> list[str]:
    """Cut one work into staff samples. Raises SkippedWork if it cannot align."""
    import fitz  # noqa: PLC0415

    # Imported here: page_notes itself builds on this module.
    from training.datasets.page_notes import (  # noqa: PLC0415
        bar_rules,
        full_size,
        key_signature,
        pair_notes,
        read_staff,
        time_signature,
        uses_usual_encoding,
        volta_hooks,
    )

    mu2_path = os.path.join(symbtr_mu2, (mu2_stem or stem) + ".mu2")
    if not os.path.exists(mu2_path):
        raise SkippedWork("no matching mu2 score")
    score = read_mu2(mu2_path)

    document = fitz.open(os.path.join(symbtr_pdf, stem + ".pdf"))
    try:
        if not any("Mus2" in font[3] for font in document[0].get_fonts(full=True)):
            raise SkippedWork("no embedded Mus2 font - notation is not machine readable")
        staves = []
        for page in document:
            glyphs = _mus2_glyphs(page)
            for staff in _staff_lines(page):
                staves.append(
                    {"page": page.number, "lines": staff, "bars": _barlines(page, staff, glyphs)}
                )
        if not staves:
            raise SkippedWork("no staves found on the page")

        measures = _cut_into_measures(score)
        # One state for the whole work: the key signature holds throughout and
        # a measure's accidentals carry on across a system break.
        score.key = printed_key_signature(document, score.key)
        state = EngravingState(score.key, printed_rounding(document))
        page_measures = sum(len(staff["bars"]) for staff in staves)
        # The final barline is often heavy or doubled and can go undetected.
        if abs(len(measures) - page_measures) > 1:
            raise SkippedWork(
                f"{len(measures)} measures in the .mu2 but {page_measures} on the page"
            )

        # The whole work as one stream of labels, then cut into staffs. The
        # engraving state runs through it in order, as the reader's eye does.
        body: list[EncodedSymbol] = []
        ends: list[int] = []
        for measure in measures:
            body.extend(_measure_tokens(measure, unresolved, state))
            ends.append(len(body))

        heads = [read_staff(document[staff["page"]], staff["lines"]) for staff in staves]

        def notes_of(start: int, end: int) -> list[tuple[str, str]]:
            return [(t.rhythm, t.pitch) for t in body[start:end] if t.rhythm.startswith("note")]

        def fits(index: int, start: int, end: int) -> bool:
            return pair_notes(notes_of(start, end), heads[index]) is not None

        cuts = _staff_cuts(body, ends, staves, [len(full_size(h)) for h in heads], fits)
        # Signs are read off the page only where its characters use the usual
        # codes; elsewhere the .mu2 reckoning stands.
        signs_readable = uses_usual_encoding(document)
        if not signs_readable:
            page_report["works whose signs cannot be read off the page"] += 1

        os.makedirs(working_dir, exist_ok=True)
        lines = []
        for index, (staff, (start, end)) in enumerate(zip(staves, cuts)):
            if start == end:
                continue
            tokens = _header(score, state, index == 0) + body[start:end]
            where = pair_notes(notes_of(start, end), heads[index])
            if where is None:
                # The page prints different notes from the ones labelled. The
                # model would be taught to see what is not there; leave it out.
                page_report["staffs left out, notes differ from the page"] += 1
                continue
            if signs_readable:
                tokens = _signature_from_page(
                    tokens, key_signature(document[staff["page"]], staff["lines"], heads[index])
                )
                if index == 0:
                    tokens = _time_from_page(
                        tokens, time_signature(document[staff["page"]], staff["lines"], heads[index])
                    )
                tokens = _signs_from_page(tokens, where, heads[index])
                tokens = _splits_from_page(tokens, where, heads[index])
                page = document[staff["page"]]
                marks = bar_rules(page, staff["lines"], heads[index])
                hooks = volta_hooks(page, staff["lines"])
                tokens = _bars_from_page(tokens, where, heads[index], marks, hooks, page.rect.width)
            page_report["staffs kept"] += 1
            base = os.path.join(working_dir, f"{stem}-{index:02d}")
            page = document[staff["page"]]
            if not just_token_files:
                lines_y = staff["lines"]
                clip = fitz.Rect(0, lines_y[0] - 34, page.rect.width, lines_y[-1] + 34)
                dpi = round(target_page_width / page.rect.width * 72)
                page.get_pixmap(clip=clip, dpi=dpi).save(base + ".png")
            with open(base + ".tokens", "w", encoding="utf-8") as token_file:
                for token in tokens:
                    token_file.write(str(token) + "\n")
            lines.append(_index_line(base))
        return lines
    finally:
        document.close()


# --- splitting --------------------------------------------------------------

_STAFF_SUFFIX = re.compile(r"-\d+\.(?:png|tokens)$")


def _work_of(line: str) -> str:
    return _STAFF_SUFFIX.sub("", os.path.basename(line.strip().split(",")[0]))


def _lift_counts(line: str, by_context: bool = False) -> collections.Counter:
    """How many of each drawn accidental one staff carries.

    With by_context the key is the sign together with where it stands -- in the
    key signature or among the notes -- which is the grain the model learns at.
    """
    token_path = git_root / line.strip().split(",")[1]
    if not token_path.exists():
        return collections.Counter()
    counts: collections.Counter = collections.Counter()
    for token_line in token_path.read_text(encoding="utf-8").splitlines():
        parts = token_line.split()
        if len(parts) == 5 and parts[2] not in (".", "_"):
            if by_context:
                place = "signature" if parts[0] == key_accidental else "notes"
                counts[(parts[2], place)] += 1
            else:
                counts[parts[2]] += 1
    return counts


def _top_up(
    train: list[str],
    extra: list[str],
    rng: random.Random,
    by_context: bool,
    target: int,
) -> None:
    """Repeat scarce staffs one at a time until every key reaches target.

    Works from rarest upwards and keeps a running tally, so a staff repeated for
    one accidental also counts towards every other it carries. MAX_REPEATS bounds
    a key too scarce to reach the target at all.
    """
    per_line = {line: _lift_counts(line, by_context) for line in set(train + extra)}
    running: collections.Counter = collections.Counter()
    for line in train + extra:
        running.update(per_line[line])

    for key in sorted(running, key=lambda name: running[name]):
        if running[key] >= target:
            continue
        carriers = [line for line in train if per_line[line][key]]
        if not carriers:
            continue
        rng.shuffle(carriers)
        budget = len(carriers) * (MAX_REPEATS - 1)
        added = 0
        while running[key] < target and added < budget:
            line = carriers[added % len(carriers)]
            extra.append(line)
            running.update(per_line[line])
            added += 1
        if added:
            label = f"{key[0]} in the {key[1]}" if by_context else str(key)
            eprint(
                f"  {label}: {added} extra staffs -> {running[key]} occurrences"
                + ("" if running[key] >= target else " (ran out of staffs)")
            )


def _oversample(train: list[str], rng: random.Random) -> list[str]:
    """Top scarce signs up, then each sign where it is scarce."""
    extra: list[str] = []
    eprint(f"Topping scarce accidentals up to {RARE_LIFT_TARGET} occurrences:")
    _top_up(train, extra, rng, by_context=False, target=RARE_LIFT_TARGET)
    eprint(f"Topping each up to {RARE_CONTEXT_TARGET} where it stands:")
    _top_up(train, extra, rng, by_context=True, target=RARE_CONTEXT_TARGET)
    return extra


def _lifts_in(line: str) -> set[str]:
    token_path = git_root / line.strip().split(",")[1]
    if not token_path.exists():
        return set()
    content = token_path.read_text(encoding="utf-8")
    return {lift for lift in RARE_LIFTS if lift in content}


def split_and_balance(lines: list[str], seed: int = 0) -> dict[str, list[str]]:
    """Split by work, so no work has staffs in more than one set.

    Only a glyph too scarce to divide at all is pinned to training; everything
    else is split randomly so the test set still contains the rare glyphs and
    their recognition can actually be measured. Staffs carrying an accidental
    the training set is thin on are then repeated, up to a target count.
    """
    works: dict[str, list[str]] = {}
    for line in lines:
        works.setdefault(_work_of(line), []).append(line)

    works_with_lift: dict[str, set[str]] = collections.defaultdict(set)
    for work, work_lines in works.items():
        for line in work_lines:
            for lift in _lifts_in(line):
                works_with_lift[lift].add(work)

    pinned = {
        work
        for lift, holders in works_with_lift.items()
        if len(holders) < MIN_WORKS_TO_SPLIT
        for work in holders
    }
    if pinned:
        scarce = sorted(
            lift for lift, holders in works_with_lift.items()
            if len(holders) < MIN_WORKS_TO_SPLIT
        )
        eprint(f"Pinning {len(pinned)} works to training; too few carry {scarce}")

    splittable = sorted(work for work in works if work not in pinned)
    rng = random.Random(seed)
    rng.shuffle(splittable)

    test_count = int(len(splittable) * TEST_FRACTION)
    val_count = int(len(splittable) * VAL_FRACTION)
    test_works = splittable[:test_count]
    val_works = splittable[test_count : test_count + val_count]
    train_works = splittable[test_count + val_count :] + sorted(pinned)

    def collect(names: list[str]) -> list[str]:
        return [line for name in names for line in works[name]]

    train = collect(train_works)
    train += _oversample(train, rng)
    rng.shuffle(train)
    return {"train": train, "val": collect(val_works), "test": collect(test_works)}


def _lift_histogram(lines: list[str]) -> collections.Counter:
    counts: collections.Counter = collections.Counter()
    for line in lines:
        token_path = git_root / line.strip().split(",")[1]
        if not token_path.exists():
            continue
        for token_line in token_path.read_text(encoding="utf-8").splitlines():
            parts = token_line.split()
            if len(parts) == 5 and parts[2] not in (".", "_", "N"):
                counts[parts[2]] += 1
    return counts


def previous_split() -> dict[str, str]:
    """Which split each work went to last time, read from the index files."""
    placed = {}
    for name, path in (("train", index_train), ("val", index_val), ("test", index_test)):
        if os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        placed[_work_of(line)] = name
    return placed


def split_as_before(lines: list[str], placed: dict[str, str], seed: int = 0) -> dict[str, list[str]]:
    """Keep every work in the split it was in, so two datasets can be compared.

    A test set that changed its works along with its labels would measure the
    new works, not the new labels. A work that was not there before goes to
    training.
    """
    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for line in lines:
        splits[placed.get(_work_of(line), "train")].append(line)
    rng = random.Random(seed)
    splits["train"] += _oversample(splits["train"], rng)
    rng.shuffle(splits["train"])
    return splits


def write_splits(lines: list[str], placed: dict[str, str] | None = None) -> None:
    splits = split_as_before(lines, placed) if placed else split_and_balance(lines)
    for name, path in (("train", index_train), ("val", index_val), ("test", index_test)):
        with open(path, "w", encoding="utf-8") as handle:
            handle.writelines(splits[name])
        eprint(f"{name}: {len(splits[name])} staff samples -> {path}")
    # Per-class counts, so a class that is missing from the test set is visible
    # here rather than after a training run.
    histograms = {name: _lift_histogram(rows) for name, rows in splits.items()}
    every_lift = sorted({lift for counts in histograms.values() for lift in counts})
    eprint(f"{'accidental':<12}{'train':>9}{'val':>7}{'test':>7}")
    for lift in every_lift:
        eprint(
            f"{lift:<12}{histograms['train'][lift]:>9}"
            f"{histograms['val'][lift]:>7}{histograms['test'][lift]:>7}"
        )


# --- entry point ------------------------------------------------------------


def convert_symbtr(
    just_token_files: bool = False,
    limit: int | None = None,
    placed: dict[str, str] | None = None,
) -> list[str]:
    if not os.path.isdir(symbtr_pdf):
        eprint(f"SymbTr pdf folder not found at {symbtr_pdf}")
        sys.exit(1)
    stems = sorted(path.stem for path in Path(symbtr_pdf).glob("*.pdf"))
    if limit:
        stems = stems[:limit]
    mu2_index = build_mu2_index()

    lines: list[str] = []
    unresolved: collections.Counter = collections.Counter()
    skipped: collections.Counter = collections.Counter()
    skipped_works: list[str] = []
    for number, stem in enumerate(stems, start=1):
        try:
            lines.extend(
                convert_work(stem, just_token_files, unresolved, mu2_index.get(stem))
            )
        except SkippedWork as reason:
            skipped[_reason_kind(str(reason))] += 1
            skipped_works.append(f"{stem}\t{reason}")
        except Exception as error:  # noqa: BLE001
            skipped[f"error: {type(error).__name__}"] += 1
            skipped_works.append(f"{stem}\t{error}")
        if number % 100 == 0:
            eprint(f"Processed {number}/{len(stems)}, {len(lines)} staffs so far")

    converted = len(stems) - sum(skipped.values())
    eprint(f"Converted {converted}/{len(stems)} works into {len(lines)} staff samples")
    for reason, count in skipped.most_common():
        eprint(f"  skipped {count}: {reason}")
    if unresolved:
        eprint(f"  durations that could not be written: {dict(unresolved.most_common(6))}")
    eprint("What the page changed:")
    for what, count in sorted(page_report.items(), key=lambda kv: (kv[0].startswith("sign"), -kv[1])):
        eprint(f"  {count:>7}  {what}")

    os.makedirs(working_dir, exist_ok=True)
    with open(os.path.join(working_dir, "skipped.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(skipped_works))
    with open(index_file, "w", encoding="utf-8") as handle:
        handle.writelines(lines)
    eprint(f"Wrote index with {len(lines)} entries to {index_file}")
    write_splits(lines, placed)
    return lines


def _reason_kind(reason: str) -> str:
    """Collapse standalone counts out of a skip reason so they group in the report."""
    return re.sub(r"\b\d+\b", "N", reason)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert SymbTr into homr training data")
    parser.add_argument(
        "--only-tokens",
        action="store_true",
        help="Only (re)generate token files, don't render staff images.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Process only the first N works."
    )
    parser.add_argument(
        "--split-only",
        action="store_true",
        help="Only re-run the train/val/test split on the existing index.txt.",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Rebuild index.txt and the splits from the already converted staff files.",
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Write the staff files here instead of datasets/SymbTr-work, leaving "
             "an earlier conversion where it is.",
    )
    parser.add_argument(
        "--keep-split",
        action="store_true",
        help="Put every work in the split the current index files put it in.",
    )
    parser.add_argument(
        "--add-train",
        action="append",
        default=[],
        help="An index of extra staffs, such as the courtesy-accidental ones, to "
             "train on besides the corpus. May be repeated. Used with --split-only.",
    )
    args = parser.parse_args()
    random.seed(0)
    placed = previous_split() if args.keep_split else None
    if args.work_dir:
        # The index files are about to be rewritten to point at the new folder;
        # keep the old ones beside them, named after the folder they point at.
        before = os.path.basename(working_dir)
        if os.path.exists(index_file):
            with open(index_file, encoding="utf-8") as handle:
                first = handle.readline().split(",")[0]
            before = first.split("/")[1] if first.count("/") >= 2 else before
        for path in (index_file, index_train, index_val, index_test):
            kept = path[: -len(".txt")] + f"_{before}.txt"
            if os.path.exists(path) and not os.path.exists(kept):
                shutil.copy(path, kept)
                eprint(f"Kept the previous {os.path.basename(path)} as {os.path.basename(kept)}")
        working_dir = os.path.join(dataset_root, args.work_dir)
    if args.reindex:
        rebuilt = reindex()
        with open(index_file, "w", encoding="utf-8") as handle:
            handle.writelines(rebuilt)
        eprint(f"Rebuilt index with {len(rebuilt)} entries from {working_dir}")
        write_splits(rebuilt, placed)
    elif args.split_only:
        with open(index_file, encoding="utf-8") as handle:
            corpus = handle.readlines()
        for extra_index in args.add_train:
            # Their names are not works of the corpus, so split_as_before puts
            # them in training -- which is also the only place they may go.
            with open(extra_index, encoding="utf-8") as handle:
                extra = [line for line in handle if line.strip()]
            eprint(f"Adding {len(extra)} staffs from {extra_index} to training")
            corpus += extra
        write_splits(corpus, placed)
    else:
        convert_symbtr(just_token_files=args.only_tokens, limit=args.limit, placed=placed)
