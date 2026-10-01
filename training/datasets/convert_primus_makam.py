"""Camera-PrIMuS incipits in the makam label conventions, as replay data for the makam fine-tuning.

PrIMuS gives every incipit twice. The semantic encoding has sounding pitches -- under a G major signature an
F with no sign drawn is "F#4" -- and the agnostic encoding lists what is printed, symbol by symbol, at its
staff position. The makam labels want the printed sign, so the two are read side by side: the n-th note of
one is the n-th note of the other, and the accidental printed just before a note in the agnostic list is
that note's lift. homr's own conversion (convert_primus) keeps the sounding pitch instead, writes the
signature as keySignature_<fifths> and the meter without its numerator, and closes every incipit with a
barline whether one is printed or not; none of that matches the makam labels.

    homr (convert_primus)                    here
    lift: sounding, kept through the bar     lift: the sign printed -- sharp4, flat5, N, ##, bb or _
    keySignature_<fifths>                    one keyAccidental row per printed sign, at its position
    timeSignature/<beat type>                timeSignature_<beats>/<beat type> (C as 4/4, cut C as 2/2)
    tie as its own symbol                    tieStart / tieStop on the two notes
    a closing barline always                 a barline only where one is printed

Only treble-clef incipits without tuplets are kept. Those that do not end on a barline use the
camera-distorted picture half the time; the rest the clean one; all get homr's random margins.

    python -m training.datasets.convert_primus_makam
"""

import hashlib
import os
import random
import re
import sys
from pathlib import Path

import cv2

from homr.simple_logging import eprint
from homr.transformer.vocabulary import EncodedSymbol, build_rhythm, empty, key_accidental
from training.transformer.image_utils import add_margin
from training.transformer.training_vocabulary import token_lines_to_str

script_location = os.path.dirname(os.path.realpath(__file__))
git_root = Path(script_location).parent.parent.absolute()
dataset_root = os.path.join(git_root, "datasets")
corpus = os.path.join(dataset_root, "Corpus")
out_root = os.path.join(dataset_root, "primus-makam")

_durations = {"whole": "1", "half": "2", "quarter": "4", "eighth": "8", "sixteenth": "16",
              "thirty_second": "32", "sixty_fourth": "64"}
_signs = {"sharp": "sharp4", "flat": "flat5", "natural": "N", "doublesharp": "##", "doubleflat": "bb"}
_steps = "CDEFGAB"
_rhythms = set(build_rhythm())


class Skip(Exception):
    pass


def treble_pitch(position: str) -> str:
    """Staff position (L1 = bottom line, S1 = the space above it, S0 below it) as a treble-clef pitch."""
    kind, number = position[0], int(position[1:])
    steps_above_e4 = 2 * (number - 1) + (1 if kind == "S" else 0)
    index = _steps.index("E") + 4 * 7 + steps_above_e4
    return f"{_steps[index % 7]}{index // 7}"


def _agnostic(token: str) -> tuple[str, str]:
    """'accidental.sharp-L5' -> ('accidental.sharp', 'L5')."""
    match = re.match(r"(.+)-([LS]-?\d+)$", token)
    if not match:
        return token, ""
    return match.group(1), match.group(2)


def _duration(text: str, grace: bool) -> str:
    dots = text.count(".")
    base = text.replace(".", "")
    if base not in _durations:
        raise Skip(f"duration {base}")
    return _durations[base] + ("G" if grace else "") + "." * dots


def convert(semantic: list[str], agnostic: list[str]) -> list[EncodedSymbol]:
    if [t for t in semantic if t.startswith("clef-")] != ["clef-G2"]:
        raise Skip("not a single treble clef")
    printed = [_agnostic(t) for t in agnostic]

    # The signature: the accidentals printed between the clef and the meter or the first note.
    signature: list[EncodedSymbol] = []
    body_start = 0
    for i, (name, position) in enumerate(printed):
        if name.startswith("clef"):
            continue
        if name.startswith("accidental."):
            sign = _signs.get(name.split(".", 1)[1])
            if sign is None:
                raise Skip(name)
            signature.append(EncodedSymbol(key_accidental, treble_pitch(position), sign, empty, "upper"))
            continue
        body_start = i
        break

    # The sign printed before each note, in note order; a printed tuplet number has no place in the labels.
    drawn: list[str] = []
    pending = empty
    seen_note = False
    for name, _position in printed[body_start:]:
        if name.startswith("accidental."):
            sign = _signs.get(name.split(".", 1)[1])
            if sign is None:
                raise Skip(name)
            pending = sign
        elif name.startswith(("note.", "gracenote.")):
            drawn.append(pending)
            pending = empty
            seen_note = True
        elif name.startswith("digit") and seen_note:
            raise Skip("tuplet or multirest number")
    notes = [t for t in semantic if t.startswith(("note-", "gracenote-"))]
    if len(notes) != len(drawn):
        raise Skip("note counts differ")

    result: list[EncodedSymbol] = [EncodedSymbol("clef_G2", empty, empty, empty, "upper"), *signature]
    note_index = 0
    tie_next = False
    for token in semantic:
        if token.startswith("clef-") or token.startswith("keySignature-"):
            continue
        if token.startswith("timeSignature-"):
            meter = token.split("-", 1)[1]
            meter = {"C": "4/4", "C/": "2/2"}.get(meter, meter)
            symbol = EncodedSymbol(f"timeSignature_{meter}")
            if symbol.rhythm not in _rhythms:
                raise Skip(symbol.rhythm)
            result.append(symbol)
        elif token.startswith(("note-", "gracenote-")):
            body = token.split("-", 1)[1]
            fermata = body.endswith("_fermata")
            body = body.removesuffix("_fermata")
            pitch_text, duration = body.split("_", 1)
            match = re.match(r"([A-G])[b#N]*(\d)$", pitch_text)
            if not match:
                raise Skip(pitch_text)
            parts = ["tieStop"] if tie_next else []
            if fermata:
                parts.append("fermata")
            rhythm = "note_" + _duration(duration, token.startswith("gracenote-"))
            result.append(EncodedSymbol(rhythm, match.group(1) + match.group(2), drawn[note_index],
                                        "_".join(sorted(parts)) or empty, "upper"))
            note_index += 1
            tie_next = False
        elif token.startswith("rest-"):
            body = token.split("-", 1)[1]
            fermata = body.endswith("_fermata")
            rhythm = "rest_" + _duration(body.removesuffix("_fermata"), False)
            result.append(EncodedSymbol(rhythm, empty, empty, "fermata" if fermata else empty, "upper"))
        elif token.startswith("multirest-"):
            count = int(token.split("-", 1)[1])
            if not 2 <= count <= 10:
                raise Skip("multirest")
            result.append(EncodedSymbol(f"rest_{count}m", empty, empty, empty, "upper"))
        elif token == "barline":
            result.append(EncodedSymbol("barline"))
        elif token == "tie":
            last = next((s for s in reversed(result) if s.rhythm.startswith("note")), None)
            if last is None:
                raise Skip("tie before any note")
            parts = sorted({*(p for p in last.articulation.split("_") if p and p != empty), "tieStart"})
            last.articulation = "_".join(parts)
            tie_next = True
        else:
            raise Skip(token)
    if any(s.rhythm not in _rhythms for s in result):
        raise Skip("token outside the vocabulary")
    return result


def main() -> None:
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    os.makedirs(out_root, exist_ok=True)
    samples = sorted(os.listdir(corpus))[:limit]
    skipped: dict[str, int] = {}
    kept = 0
    with open(os.path.join(out_root, "index.txt"), "w", encoding="utf-8", newline="\n") as index:
        for i, name in enumerate(samples, 1):
            folder = os.path.join(corpus, name)
            try:
                with open(os.path.join(folder, name + ".semantic"), encoding="utf-8", errors="ignore") as f:
                    semantic = f.read().split()
                with open(os.path.join(folder, name + ".agnostic"), encoding="utf-8", errors="ignore") as f:
                    agnostic = f.read().split()
                tokens = convert(semantic, agnostic)
            except Skip as e:
                reason = str(e).split(" ")[0]
                skipped[reason] = skipped.get(reason, 0) + 1
                continue
            # The camera pictures are cropped tight, and a closing barline on the right edge is often cut
            # off or too faint to see; an incipit that ends on one takes the clean picture.
            camera = int(hashlib.md5(name.encode()).hexdigest(), 16) % 2 == 0 and tokens[-1].rhythm != "barline"  # noqa: S324
            source = os.path.join(folder, name + ("_distorted.jpg" if camera else ".png"))
            image = cv2.imread(source, cv2.IMREAD_GRAYSCALE)
            if image is None:
                skipped["image"] = skipped.get("image", 0) + 1
                continue
            rng = random.Random(name)
            image = add_margin(image, rng.randint(10, 30), rng.randint(10, 30), rng.randint(0, 10),
                               rng.randint(0, 10))
            png = os.path.join(out_root, name + ".png")
            tok = os.path.join(out_root, name + ".tokens")
            cv2.imwrite(png, image)
            with open(tok, "w", encoding="utf-8", newline="\n") as f:
                f.write(token_lines_to_str(tokens))
            index.write(f"{Path(png).relative_to(git_root).as_posix()},{Path(tok).relative_to(git_root).as_posix()}\n")
            kept += 1
            if i % 5000 == 0:
                eprint(f"{i}/{len(samples)} incipits, {kept} kept")
    eprint(f"Done: {kept} of {len(samples)} kept; skipped {sorted(skipped.items(), key=lambda x: -x[1])}")


if __name__ == "__main__":
    main()
