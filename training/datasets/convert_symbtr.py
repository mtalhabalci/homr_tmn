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
OVERSAMPLE_FACTOR = 8
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
    pending_marks: tuple[str | None, str | None] | None = None
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) < 9:
            continue
        code = fields[0].strip()
        if code == KEY_SIGNATURE_CODE:
            key = fields[8].strip()
            continue
        opening, closing = fields[7].strip(), fields[8].strip()
        if opening in OPENING_MARKS or closing in CLOSING_MARKS:
            pending_marks = (OPENING_MARKS.get(opening), CLOSING_MARKS.get(closing))
            continue
        if code in GRACE_CODES:
            events.append({"name": fields[1].strip(), "duration": Fraction(0), "grace": True})
            continue
        if code not in NOTE_CODES:
            continue
        numerator, denominator = fields[2].strip(), fields[3].strip()
        if not (numerator.isdigit() and denominator.isdigit()):
            continue
        if not int(numerator) or not int(denominator):
            continue
        event = {
            "name": fields[1].strip(),
            "duration": Fraction(int(numerator), int(denominator)),
            "grace": False,
        }
        if pending_marks:
            event["open"], event["close"] = pending_marks
            pending_marks = None
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

    def __init__(self, key: str):
        self.signature: dict[str, int] = {}
        for entry in (part.strip() for part in key.split("/")):
            parsed = parse_note_name(entry) if entry else None
            if parsed:
                self.signature[parsed[0][0]] = parsed[1]
        self.measure: dict[str, int] = {}

    def start_measure(self) -> None:
        self.measure = {}

    def drawn_lift(self, pitch: str, commas: int) -> str:
        """The accidental printed before this note, or empty if there is none."""
        letter, expected = pitch[0], self.measure.get(pitch)
        if expected is None:
            expected = self.signature.get(letter, 0)
        if commas == expected:
            return empty
        self.measure[pitch] = commas
        return lift_for_commas(commas)


def tokens_for_event(
    event: dict, unresolved: collections.Counter, state: EngravingState | None = None
) -> list[EncodedSymbol]:
    pitch = parse_note_name(event["name"])
    if event["grace"]:
        parts, suffix = [(8, 0)], "G"
    else:
        parts, suffix = split_duration(event["duration"]), ""
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


def _staff_tokens(
    measures: list[list[dict]],
    score: Mu2Score,
    is_first_staff: bool,
    unresolved: collections.Counter,
    state: EngravingState,
) -> list[EncodedSymbol]:
    # Mus2 redraws the clef and the whole key signature at the head of every
    # system, so every staff sample carries them; only the time signature is
    # printed once, at the start of the piece. The staff image is all the model
    # ever sees, so the labels have to match what is on that one staff.
    tokens: list[EncodedSymbol] = [EncodedSymbol("clef_G2", "_", "_", "_", "upper")]
    for entry in (part.strip() for part in score.key.split("/")):
        pitch = parse_note_name(entry) if entry else None
        if pitch:
            tokens.append(
                EncodedSymbol(key_accidental, pitch[0], lift_for_commas(pitch[1]), "_", "upper")
            )
    if is_first_staff:
        tokens.append(EncodedSymbol(f"timeSignature_{score.numerator}/{score.denominator}"))
    for measure in measures:
        opening = next((e["open"] for e in measure if e.get("open")), None)
        closing = next((e["close"] for e in measure if e.get("close")), None)
        if opening:
            tokens.append(EncodedSymbol(opening))
        state.start_measure()
        for event in measure:
            tokens.extend(tokens_for_event(event, unresolved, state))
        tokens.append(EncodedSymbol(closing or "barline"))
    return tokens


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
        state = EngravingState(score.key)
        page_measures = sum(len(staff["bars"]) for staff in staves)
        # The final barline is often heavy or doubled and can go undetected.
        if abs(len(measures) - page_measures) > 1:
            raise SkippedWork(
                f"{len(measures)} measures in the .mu2 but {page_measures} on the page"
            )

        os.makedirs(working_dir, exist_ok=True)
        lines = []
        taken = 0
        for index, staff in enumerate(staves):
            is_last = index == len(staves) - 1
            mine = (
                measures[taken:]
                if is_last
                else measures[taken : taken + len(staff["bars"])]
            )
            taken += len(mine)
            tokens = _staff_tokens(mine, score, index == 0, unresolved, state)
            if not tokens:
                continue
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
    their recognition can actually be measured. Rare staffs are then repeated
    within training.
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
    train += [line for line in train for _ in range(OVERSAMPLE_FACTOR - 1) if _lifts_in(line)]
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


def write_splits(lines: list[str]) -> None:
    splits = split_and_balance(lines)
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


def convert_symbtr(just_token_files: bool = False, limit: int | None = None) -> list[str]:
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

    os.makedirs(working_dir, exist_ok=True)
    with open(os.path.join(working_dir, "skipped.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(skipped_works))
    with open(index_file, "w", encoding="utf-8") as handle:
        handle.writelines(lines)
    eprint(f"Wrote index with {len(lines)} entries to {index_file}")
    write_splits(lines)
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
    args = parser.parse_args()
    random.seed(0)
    if args.reindex:
        rebuilt = reindex()
        with open(index_file, "w", encoding="utf-8") as handle:
            handle.writelines(rebuilt)
        eprint(f"Rebuilt index with {len(rebuilt)} entries from {working_dir}")
        write_splits(rebuilt)
    elif args.split_only:
        with open(index_file, encoding="utf-8") as handle:
            write_splits(handle.readlines())
    else:
        convert_symbtr(just_token_files=args.only_tokens, limit=args.limit)
