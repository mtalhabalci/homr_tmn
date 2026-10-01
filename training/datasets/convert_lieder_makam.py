"""Lieder voice staffs in the makam label conventions, as replay data for the makam fine-tuning.

homr learned Western engraving from the OpenScore Lieder corpus. Fine-tuning on makam pages alone
lets that fade -- slurs, ties, staccato, accents and fermatas above all, which most makam labels
leave out -- so a share of these staffs goes back into the mix. They cannot go in as homr labels
them, because homr's labels follow other conventions than ours:

    homr (convert_lieder)                        here
    lift: the sounding alteration, naturals cut   lift: the sign drawn -- sharp4, flat5, N, ##, bb or _
    keySignature_<fifths>                         one keyAccidental row per sign, left to right
    timeSignature/<beat type>                     timeSignature_<beats>/<beat type>
    voltas, segno, coda, D.C. dropped             works that draw them are left out: a drawn mark
                                                  without its token would teach the model to skip it

A plain sharp is sharp4 and a plain flat flat5 because that is what the makam labels call the same
drawn signs; what the sign means is settled after reading, not in the label.

Only voice staffs are kept: one staff, treble clef, no chords -- the shape of a makam staff, with the
lyrics under it as in a sarki. Windows cannot run MuseScore's batch job (-j), so every score is
exported on its own, and the pages are rasterised with resvg instead of rsvg-convert.

    python -m training.datasets.convert_lieder_makam [--render-only] [--limit N]
"""

import glob
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from xml.dom import minidom

import cv2
import numpy as np
import resvg_py

from homr.simple_logging import eprint
from homr.transformer.vocabulary import (
    EncodedSymbol,
    build_articulation,
    build_rhythm,
    empty,
    key_accidental,
)
from training.datasets.convert_lieder import MeasureCutter, is_grandstaff
from training.datasets.music_xml_parser import music_xml_string_to_tokens
from training.datasets.musescore_svg import SvgMusicFile, SvgStaff, get_position_from_multiple_svg_files
from training.transformer.training_vocabulary import calc_ratio_of_tuplets, token_lines_to_str

script_location = os.path.dirname(os.path.realpath(__file__))
git_root = Path(script_location).parent.parent.absolute()
dataset_root = os.path.join(git_root, "datasets")
lieder_scores = os.path.join(dataset_root, "Lieder-main", "scores")
out_root = os.path.join(dataset_root, "lieder-makam")
flat = os.path.join(out_root, "flat")
musescore = r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe"
target_width = 1400

# Scores MuseScore hangs on (from convert_lieder).
known_hangs = ("lc6264558", "lc5712131", "lc5995407", "lc5712146", "lc6001354", "lc6248307", "lc5935864")

# Drawn in the picture but absent from the tokens the parser writes: a work that has any is left out.
_unlabelled_marks = ("ending", "segno", "coda", "octave-shift")
# Small cue notes are drawn but would be labelled as ordinary notes. Rather than lose every work that
# has a few, they get an articulation no vocabulary has, and to_makam drops the staffs they are on.
_cue_mark = "unstress"
_unlabelled_words = re.compile(r"\b(D\.\s*C\.|D\.\s*S\.|da\s+capo|dal\s+segno)", re.IGNORECASE)
_drawn_bar_styles = ("regular", "light-light", "light-heavy")

_lift_names = {"sharp": "sharp4", "flat": "flat5", "natural": "N", "double-sharp": "##",
               "sharp-sharp": "##", "flat-flat": "bb"}
_lift_tokens = {"#": "sharp4", "b": "flat5", "N": "N", "##": "##", "bb": "bb", empty: empty}

# Where each signature sign sits on a treble staff, in the order it is drawn.
_sharp_order = ("F5", "C5", "G5", "D5", "A4", "E5", "B4")
_flat_order = ("B4", "E5", "A4", "D5", "G4", "C5", "F4")

_rhythms = set(build_rhythm())
_articulations = set(build_articulation())


def render(mscx: str) -> None:
    """MusicXML and one SVG per page next to a copy of the score, unless already there."""
    name = os.path.basename(mscx)
    copy = os.path.join(flat, name)
    base = copy[: -len(".mscx")]
    if not os.path.exists(copy):
        shutil.copyfile(mscx, copy)
    for target, done in ((base + ".musicxml", base + ".musicxml"), (base + ".svg", base + "-1.svg")):
        if os.path.exists(done):
            continue
        try:
            subprocess.run([musescore, "-o", target, copy], timeout=180, check=False,  # noqa: S603
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            eprint("MuseScore timed out on", name)
            return


def drawn_signs_only(content: bytes) -> str | None:
    """The MusicXML rewritten so the parser reads the drawn sign as the lift, or None to skip the work.

    The parser takes the lift from <accidental> when there is one and from <alter> otherwise, so the
    <alter> of a note with no drawn sign is removed. A double sign is left to <alter> (±2), which the
    parser does read. The time signature's beats are folded into the beat type, "3x4", which the
    parser copies into its token and to_makam unfolds.
    """
    root = ET.fromstring(content)  # noqa: S314
    if any(next(root.iter(tag), None) is not None for tag in _unlabelled_marks):
        return None
    # MuseScore 4.7 writes <miscellaneous-field name=...>, an attribute the musicxml package cannot set.
    for identification in root.findall("identification"):
        root.remove(identification)
    for words in root.iter("words"):
        if words.text and _unlabelled_words.search(words.text):
            return None
    for style in root.iter("bar-style"):
        if style.text not in _drawn_bar_styles:
            return None
    for note in root.iter("note"):
        if note.find("cue") is not None:
            notations = note.find("notations")
            if notations is None:
                notations = ET.SubElement(note, "notations")
            ET.SubElement(ET.SubElement(notations, "articulations"), _cue_mark)
        pitch = note.find("pitch")
        if pitch is None:
            continue
        accidental = note.find("accidental")
        alter = pitch.find("alter")
        if accidental is None:
            if alter is not None:
                pitch.remove(alter)
            continue
        drawn = _lift_names.get((accidental.text or "").strip())
        if drawn is None:
            return None  # a sign the makam labels have no token for
        if drawn in ("##", "bb"):
            note.remove(accidental)
            if alter is None:
                alter = ET.SubElement(pitch, "alter")
            alter.text = "2" if drawn == "##" else "-2"
            # <alter> must follow <step>; ElementTree appends it after <octave>, which the parser accepts.
    for time in root.iter("time"):
        beats, beat_type = time.find("beats"), time.find("beat-type")
        if beats is None or beat_type is None:
            return None
        symbol = time.attrib.pop("symbol", "")
        if symbol in ("common", "cut"):
            beats.text, beat_type.text = ("4", "4") if symbol == "common" else ("2", "2")
        beat_type.text = f"{(beats.text or '').strip()}x{(beat_type.text or '').strip()}"
    return ET.tostring(root, encoding="unicode")


def key_rows(fifths: int) -> list[EncodedSymbol]:
    if fifths > 0:
        return [EncodedSymbol(key_accidental, p, "sharp4", empty, "upper") for p in _sharp_order[:fifths]]
    return [EncodedSymbol(key_accidental, p, "flat5", empty, "upper") for p in _flat_order[: -fifths]]


def to_makam(symbols: list[EncodedSymbol]) -> list[EncodedSymbol] | None:
    """One voice staff in the makam conventions, or None if it is not one we can label."""
    result: list[EncodedSymbol] = []
    notes = 0
    for s in symbols:
        rhythm = s.rhythm
        if rhythm == "chord" or s.position == "lower":
            return None
        if rhythm.startswith("clef"):
            if rhythm != "clef_G2":
                return None
            result.append(EncodedSymbol(rhythm, empty, empty, empty, "upper"))
        elif rhythm.startswith("keySignature_"):
            result.extend(key_rows(int(rhythm.split("_")[1])))
        elif rhythm.startswith("timeSignature/"):
            beats, _, beat_type = rhythm.split("/", 1)[1].partition("x")
            token = f"timeSignature_{beats}/{beat_type}"
            if token not in _rhythms:
                return None
            result.append(EncodedSymbol(token))
        elif rhythm.startswith(("note", "rest")):
            if s.lift not in _lift_tokens or s.articulation not in _articulations:
                return None
            if rhythm.startswith("note"):
                notes += 1
            result.append(EncodedSymbol(rhythm, s.pitch, _lift_tokens[s.lift], s.articulation, "upper"))
        else:
            result.append(EncodedSymbol(rhythm))
    if notes < 4 or any(s.rhythm not in _rhythms for s in result):
        return None
    return result


def _page_image(svg_file: SvgMusicFile) -> tuple[np.ndarray, float, float]:
    """The page as a grey image, the image pixels per SVG unit, and SVG units per CSS pixel.

    MuseScore 4.7 gives the page size in mm and draws in a finer viewBox, so homr's margins, set in
    the CSS pixels of MuseScore 4.2, are converted through the page width.
    """
    with open(svg_file.filename, encoding="utf-8") as f:
        svg = f.read()
    view_width = float(re.search(r'viewBox="[\d.\-]+ [\d.\-]+ ([\d.]+)', svg).group(1))  # type: ignore[union-attr]
    width_attr = re.search(r'<svg[^>]*\swidth="([\d.]+)(mm|px)?"', svg)
    css_width = float(width_attr.group(1)) * (96 / 25.4 if width_attr.group(2) == "mm" else 1)  # type: ignore[union-attr]
    # resvg rejects the mm page size ("invalid size"); without it the viewBox sets the size.
    svg = re.sub(r'(<svg[^>]*?)\swidth="[^"]+"\s+height="[^"]+"', r"\1", svg, count=1)
    png = bytes(resvg_py.svg_to_bytes(svg_string=svg, width=target_width, background="white"))
    image = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE)
    return image, target_width / view_width, view_width / css_width


def systems(svg_file: SvgMusicFile) -> list[list[SvgStaff]]:
    """The staffs of a page grouped into systems, top to bottom.

    homr deals the staffs out to the parts in turn, which holds only while every system shows every
    part. Lieder scores hide a staff that is empty for a whole system -- the voice during a piano
    introduction -- and from there on each staff went to the wrong part. So the systems are read from
    the page instead: the barline MuseScore draws down the left edge of a system joins its staffs.
    """
    doc = minidom.parse(svg_file.filename)  # noqa: S318
    staffs = sorted(svg_file.staffs, key=lambda s: s.y)
    joins = []
    for line in doc.getElementsByTagName("polyline"):
        if line.getAttribute("class") != "BarLine":
            continue
        (x1, y1), (x2, y2) = (map(float, p.split(",")) for p in line.getAttribute("points").split())
        if abs(x1 - x2) < 1 and any(abs(x1 - s.x) < 20 for s in staffs):
            joins.append((min(y1, y2), max(y1, y2)))
    groups: list[list[SvgStaff]] = []
    for staff in staffs:
        # Joined when one left-edge barline reaches into both this staff and the one above it.
        joined = groups and any(
            top <= groups[-1][-1].y + groups[-1][-1].height and bottom >= staff.y
            and bottom >= groups[-1][-1].y and top <= staff.y + staff.height
            for top, bottom in joins
        )
        if joined:
            groups[-1].append(staff)
        else:
            groups.append([staff])
    return groups


def measures(staff: SvgStaff, unit: float) -> int:
    """Measures on a staff, counting barlines closer than 50 CSS px as one (a double or final line).

    SvgStaff counts with 50 SVG units, right for MuseScore 4.2 whose units were CSS pixels; in 4.7's finer
    units the two strokes of a final barline are 60 apart and made one measure more.
    """
    positions = sorted(staff.bar_line_x_positions)
    kept = [positions[0]]
    for x in positions[1:]:
        if x - kept[-1] >= 50 * unit:
            kept.append(x)
        else:
            kept[-1] = x  # the staff's right end stands in for a line drawn just before it
    return len(kept) - 1


def convert_file(musicxml: str) -> list[str]:
    """Index lines "png,tokens" for the voice staffs of one work; nothing unless the whole work lines up."""
    try:
        with open(musicxml, "rb") as f:
            content = drawn_signs_only(f.read())
        if content is None:
            return []
        voices = music_xml_string_to_tokens(content)
        cutters = [MeasureCutter(v) for v in voices]
        staff_counts = [2 if is_grandstaff(v) else 1 for v in voices]
        total = len(voices[0])
        consumed = 0
        found: list[tuple[np.ndarray, list[EncodedSymbol], str]] = []
        for page_no, svg_file in enumerate(get_position_from_multiple_svg_files(musicxml), 1):
            image, scale, unit = _page_image(svg_file)
            for system_no, system in enumerate(systems(svg_file), 1):
                bars = measures(system[0], unit)
                if any(measures(s, unit) != bars for s in system):
                    return []
                shown = len(system) == sum(staff_counts)
                row = 0
                for part, (cutter, count) in enumerate(zip(cutters, staff_counts, strict=True)):
                    symbols = cutter.extract_measures(bars)
                    if shown and count == 1 and calc_ratio_of_tuplets(symbols) <= 0.2:
                        tokens = to_makam(symbols)
                        if tokens is not None:
                            area = system[row]
                            # homr's margins: 40 px left, 10 right, 50 above and below for ledger
                            # lines, slurs and the lyrics.
                            x0 = int((area.x - 40 * unit) * scale)
                            x1 = int((area.x + area.width + 10 * unit) * scale)
                            y0 = int((area.y - 50 * unit) * scale)
                            y1 = int((area.y + area.height + 50 * unit) * scale)
                            crop = image[max(0, y0) : y1, max(0, x0) : x1]
                            found.append((crop, tokens, f"-p{page_no}-s{system_no}-v{part + 1}"))
                    row += count
                consumed += bars
        if consumed != total:
            return []
    except Exception as e:  # a score the parser or the layout match cannot follow is skipped
        eprint("Skipped", os.path.basename(musicxml), type(e).__name__, e)
        return []
    base = musicxml[: -len(".musicxml")]
    lines = []
    for crop, tokens, suffix in found:
        png, tok = base + suffix + ".png", base + suffix + ".tokens"
        cv2.imwrite(png, crop)
        with open(tok, "w", encoding="utf-8", newline="\n") as f:
            f.write(token_lines_to_str(tokens))
        lines.append(f"{Path(png).relative_to(git_root).as_posix()},{Path(tok).relative_to(git_root).as_posix()}\n")
    return lines


def main() -> None:
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    os.makedirs(flat, exist_ok=True)
    scores = sorted(p for p in glob.glob(os.path.join(lieder_scores, "**", "*.mscx"), recursive=True)
                    if not any(h in p for h in known_hangs))[:limit]
    eprint("Rendering", len(scores), "scores")
    with ThreadPoolExecutor(3) as pool:
        for i, _ in enumerate(pool.map(render, scores), 1):
            if i % 50 == 0:
                eprint("rendered", i)
    if "--render-only" in sys.argv:
        return
    works = sorted(glob.glob(os.path.join(flat, "*.musicxml")))
    kept = 0
    with open(os.path.join(out_root, "index.txt"), "w", encoding="utf-8", newline="\n") as index:
        for i, musicxml in enumerate(works, 1):
            lines = convert_file(musicxml)
            kept += len(lines)
            index.writelines(lines)
            if i % 50 == 0:
                eprint(f"{i}/{len(works)} works, {kept} voice staffs")
    eprint(f"Done: {kept} voice staffs from {len(works)} works")


if __name__ == "__main__":
    main()
