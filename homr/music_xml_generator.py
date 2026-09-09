import math
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction

import musicxml.xmlelement.xmlelement as mxl
import numpy as np

from homr import constants
from homr.makam_key import resolve_sounding
from homr.simple_logging import eprint
from homr.transformer.vocabulary import (
    EncodedSymbol,
    aeu_accidentals,
    empty,
    key_accidental,
    nonote,
    sort_token_chords,
)


class ConversionState:
    def __init__(self, division: int, nominator: Fraction):
        self.beats = 4 * constants.duration_of_quarter
        self.division = division
        self.nominator = nominator
        self.tremolo_state = "stop"
        self.volta_number = 1
        self.last_volta_measure = -10

    def start_volta(self, measure_no: int) -> int:
        if measure_no == self.last_volta_measure + 1:
            self.volta_number += 1
        else:
            self.volta_number = 1
        return self.volta_number

    def stop_volta(self, measure_no: int) -> int:
        self.last_volta_measure = measure_no
        return self.volta_number

    def toggle_tremolo_state(self) -> str:
        if self.tremolo_state == "start":
            self.tremolo_state = "stop"
        else:
            self.tremolo_state = "start"
        return self.tremolo_state


class SymbolChord:
    def __init__(self, symbols: list[EncodedSymbol], tuplet_mark: str = "") -> None:
        self.symbols = symbols
        self.tuplet_mark = tuplet_mark

    def __str__(self) -> str:
        return str.join("&", [str(s) for s in self.symbols])

    def __repr__(self) -> str:
        return str(self)

    def is_barline(self) -> bool:
        if len(self.symbols) == 0:
            return False
        first_rhythm = self.symbols[0].rhythm
        return "barline" in first_rhythm or "repeat" in first_rhythm

    def get_duration(self) -> Fraction:
        notes_rests = [
            s.get_duration().fraction for s in self.symbols if s.rhythm.startswith(("note", "rest"))
        ]
        if len(notes_rests) == 0:
            return Fraction(0)
        return min(notes_rests)

    def into_positions(self) -> list["SymbolChord"]:
        upper = []
        lower = []
        lower_is_only_rest = True
        for symbol in self.symbols:
            if symbol.position == "upper":
                upper.append(symbol)
            else:
                lower.append(symbol)
                lower_is_only_rest = lower_is_only_rest and symbol.rhythm.startswith("rest")
        chords = (
            SymbolChord(upper, self.tuplet_mark),
            SymbolChord(lower, self.tuplet_mark),
        )
        if lower_is_only_rest:
            chords = (chords[1], chords[0])
        return [chord for chord in chords if len(chord.symbols) > 0]

class XmlGeneratorArguments:
    def __init__(
        self, large_page: bool | None = None, metronome: int | None = None, tempo: int | None = None
    ):
        self.large_page = large_page
        self.metronome = metronome
        self.tempo = tempo


def build_identification() -> mxl.XMLIdentification:
    """Identification/encoding so validators and apps (e.g. MuseScore) can attribute the file."""
    ident = mxl.XMLIdentification()
    enc = mxl.XMLEncoding()
    enc.add_child(mxl.XMLSoftware(value_="homr"))
    ident.add_child(enc)
    return ident


def generate_xml(
    args: XmlGeneratorArguments, staffs: list[list[EncodedSymbol]], title: str
) -> mxl.XMLElement:
    root = mxl.XMLScorePartwise(version="4.0")
    root.add_child(build_work(title))
    root.add_child(build_identification())
    root.add_child(build_defaults(args))
    has_two_staves_by_part = [_voice_has_two_staves(staff) for staff in staffs]
    root.add_child(build_part_list(has_two_staves_by_part))
    for index, staff in enumerate(staffs):
        root.add_child(
            build_part(args, resolve_sounding(staff), index, has_two_staves_by_part[index])
        )
    return root


def _voice_has_two_staves(voice: list[EncodedSymbol]) -> bool:
    """True if any symbol uses the lower staff (e.g. piano left hand / bass clef)."""
    return any(s.position == "lower" for s in voice)


def build_part(
    args: XmlGeneratorArguments, voice: list[EncodedSymbol], index: int, has_two_staves: bool
) -> mxl.XMLPart:
    part = mxl.XMLPart(id=get_part_id(index))
    is_first_part = index == 0
    measures = build_measures(args, voice, is_first_part, has_two_staves)
    for measure in measures:
        part.add_child(measure)
    return part


def build_measures(
    args: XmlGeneratorArguments,
    voice: list[EncodedSymbol],
    is_first_part: bool,
    has_two_staves: bool = False,
) -> list[mxl.XMLMeasure]:
    def close_current_measure() -> None:
        rebalance_measure_voices(current_measure)
        measures.append(current_measure)

    measure_number = 1
    groups = add_tuplet_start_stop(group_into_chords(voice))
    division, nominator = find_division_and_time_signature_nominator(groups)
    state = ConversionState(division, nominator)
    measures: list[mxl.XMLMeasure] = []
    current_measure = mxl.XMLMeasure(number=str(measure_number))
    first_attributes = build_or_get_attributes(current_measure, None)
    first_attributes.add_child(build_divisions(division))
    if has_two_staves:
        first_attributes.add_child(mxl.XMLStaves(value_=2))
        first_attributes.add_child(mxl.XMLPartSymbol(value_="brace"))
    if is_first_part:
        direction = build_add_time_direction(args)
        if direction:
            current_measure.add_child(direction)
    attributes: mxl.XMLAttributes | None = first_attributes
    for group_no, group in enumerate(groups):
        symbol = group.symbols[0]
        rhythm = symbol.rhythm
        last_attributes = attributes
        attributes = None
        if rhythm.startswith(("note", "rest")):
            if len(group.symbols) == 1 and rhythm.endswith("m"):
                attributes = build_or_get_attributes(current_measure, last_attributes)
                build_multi_measure_rest(symbol, attributes)
            else:
                staff_positions = group.into_positions()
                for pos_no, staff_pos in enumerate(staff_positions):
                    chord_duration = (
                        group.get_duration() if pos_no == len(staff_positions) - 1 else Fraction(0)
                    )
                    for note_xml in build_note_chord(staff_pos, state, chord_duration):
                        current_measure.add_child(note_xml)
            continue
        if rhythm == "newline":
            is_last_measure = group_no == len(groups) - 1
            if not is_last_measure:
                current_measure.add_child(mxl.XMLPrint(new_system="yes"))
        elif rhythm.startswith("clef"):
            attributes = build_or_get_attributes(current_measure, last_attributes, force_new=True)
            for should_be_clef in group.symbols:
                if should_be_clef.rhythm.startswith("clef"):
                    build_clef(should_be_clef, attributes)
        elif rhythm.startswith("keySignature"):
            attributes = build_or_get_attributes(current_measure, last_attributes)
            build_key(symbol, attributes)
        elif rhythm == key_accidental:
            # A makam signature arrives as one keyAccidental symbol per
            # accidental. Collect the whole consecutive run into a single <key>,
            # and let the later symbols of that run fall through as no-ops.
            previous = groups[group_no - 1].symbols[0].rhythm if group_no > 0 else ""
            if previous != key_accidental:
                run = []
                for later in groups[group_no:]:
                    if later.symbols[0].rhythm != key_accidental:
                        break
                    run.append(later.symbols[0])
                attributes = build_or_get_attributes(current_measure, last_attributes)
                build_makam_key(run, attributes)
            else:
                attributes = last_attributes
        elif rhythm.startswith("timeSignature"):
            attributes = build_or_get_attributes(current_measure, last_attributes)
            build_time_signature(symbol, attributes, state)
        elif "barline" in rhythm:
            if rhythm != "barline":
                # Standard barlines don't need extra handling
                barline = build_or_get_barline(current_measure, "right")
                build_barline_style(symbol, barline)

            close_current_measure()
            measure_number += 1
            current_measure = mxl.XMLMeasure(number=str(measure_number))
        elif rhythm == "repeatStart":
            close_current_measure()
            measure_number += 1
            current_measure = mxl.XMLMeasure(number=str(measure_number))

            barline = build_or_get_barline(current_measure, "right")
            build_repeat(symbol, barline)
        elif rhythm == "repeatEnd":
            barline = build_or_get_barline(current_measure, "right")
            build_repeat(symbol, barline)

            close_current_measure()
            measure_number += 1
            current_measure = mxl.XMLMeasure(number=str(measure_number))
        elif rhythm == "repeatEndStart":
            barline = build_or_get_barline(current_measure, "right")
            build_repeat(EncodedSymbol("repeatEnd"), barline)

            close_current_measure()
            measure_number += 1
            current_measure = mxl.XMLMeasure(number=str(measure_number))
            barline = build_or_get_barline(current_measure, "right")
            build_repeat(EncodedSymbol("repeatStart"), barline)
        elif rhythm.startswith("voltaStart"):
            volta_number = state.start_volta(measure_number)
            barline = build_or_get_barline(current_measure, "left")
            build_barline_ending(symbol, barline, volta_number)
        elif rhythm.startswith(("voltaStop", "voltaDiscontinue")):
            volta_number = state.stop_volta(measure_number)
            # Two conventions meet here. A MusicXML source names the barline and
            # then the ending on it, so by now the measure has already closed and
            # the ending belongs to the one before. A Mus2 score draws the ending
            # bracket as the barline, so nothing has closed yet and this symbol
            # is what closes it -- without that, the next ending landed in the
            # same measure and MusicXML allows a barline only one.
            if current_measure.get_children():
                barline = build_or_get_barline(current_measure, "right")
                build_barline_ending(symbol, barline, volta_number)
                close_current_measure()
                measure_number += 1
                current_measure = mxl.XMLMeasure(number=str(measure_number))
            elif measures:
                barline = build_or_get_barline(measures[-1], "right")
                build_barline_ending(symbol, barline, volta_number)
        else:
            eprint("Symbol isn't supported yet ", symbol)

    if len(current_measure.get_children()) > 0:
        close_current_measure()
    return measures


def build_work(title_text: str) -> mxl.XMLWork:
    work = mxl.XMLWork()
    title = mxl.XMLWorkTitle()
    title._value = title_text
    work.add_child(title)
    return work


def build_defaults(args: XmlGeneratorArguments) -> mxl.XMLDefaults:
    if not args.large_page:
        return mxl.XMLDefaults()
    # These values are larger than a letter or A4 format so that
    # we only have to break staffs with every new detected staff
    # This works well for electronic formats, if the results are supposed
    # to get printed then they might need to be scaled down to fit the page
    page_width = 110  # Unit is in tenths: https://www.w3.org/2021/06/musicxml40/musicxml-reference/elements/page-height/
    page_height = 300
    defaults = mxl.XMLDefaults()
    page_layout = mxl.XMLPageLayout()
    page_height = mxl.XMLPageHeight(value_=page_height)
    page_width = mxl.XMLPageWidth(value_=page_width)
    page_layout.add_child(page_height)
    page_layout.add_child(page_width)
    defaults.add_child(page_layout)
    return defaults


def get_part_id(index: int) -> str:
    return "P" + str(index + 1)


def _part_metadata(has_two_staves: bool) -> tuple[str, str, str, int]:
    """Return (part_name, instrument_name, instrument_sound, midi_program) for a part.

    We classify by staff layout only:
    - single-staff parts -> Voice
    - two-staff parts -> Piano
    midi_program is 1-based (MusicXML: 1-128; 1 = Acoustic Grand Piano, 54 = Voice Oohs).
    """
    if has_two_staves:
        return ("Piano", "Piano", "keyboard.piano", 1)
    return ("Voice", "Voice", "voice", 54)


def build_part_list(has_two_staves_by_part: list[bool]) -> mxl.XMLPartList:
    part_list = mxl.XMLPartList()
    for part, has_two_staves in enumerate(has_two_staves_by_part):
        part_id = get_part_id(part)
        part_name_str, instrument_name_str, instrument_sound_str, midi_program = _part_metadata(
            has_two_staves
        )
        score_part = mxl.XMLScorePart(id=part_id)
        part_name = mxl.XMLPartName(value_=part_name_str)
        score_part.add_child(part_name)
        score_instrument = mxl.XMLScoreInstrument(id=part_id + "-I1")
        instrument_name = mxl.XMLInstrumentName(value_=instrument_name_str)
        score_instrument.add_child(instrument_name)
        instrument_sound = mxl.XMLInstrumentSound(value_=instrument_sound_str)
        score_instrument.add_child(instrument_sound)
        score_part.add_child(score_instrument)
        midi_instrument = mxl.XMLMidiInstrument(id=part_id + "-I1")
        midi_instrument.add_child(mxl.XMLMidiChannel(value_=part + 1))
        midi_instrument.add_child(mxl.XMLMidiProgram(value_=midi_program))
        midi_instrument.add_child(mxl.XMLVolume(value_=100))
        midi_instrument.add_child(mxl.XMLPan(value_=0))
        score_part.add_child(midi_instrument)
        part_list.add_child(score_part)
    return part_list


def build_or_get_attributes(
    measure: mxl.XMLMeasure, last_attributes: mxl.XMLAttributes | None, force_new: bool = False
) -> mxl.XMLAttributes:
    if last_attributes is not None and not force_new:
        return last_attributes

    attributes = mxl.XMLAttributes()
    measure.add_child(attributes)
    return attributes


def build_or_get_barline(measure: mxl.XMLMeasure, location: str) -> mxl.XMLBarline:
    children = measure.get_children_of_type(mxl.XMLBarline)
    for child in children:
        if child.location == location:
            return child

    barline = mxl.XMLBarline(location=location)
    measure.add_child(barline)
    return barline


def build_key(model_key: EncodedSymbol, attributes: mxl.XMLAttributes) -> None:
    key = mxl.XMLKey()
    circle_of_fifth = model_key.rhythm.split("_")[1]
    fifth = mxl.XMLFifths(value_=int(circle_of_fifth))
    attributes.add_child(key)
    key.add_child(fifth)


def build_makam_key(
    accidentals: list[EncodedSymbol], attributes: mxl.XMLAttributes
) -> None:
    """Write a makam key signature as a MusicXML non-traditional key.

    A makam signature is a list of (pitch, accidental) pairs rather than a
    position on the circle of fifths, which is what <key-step> is for. We always
    write <key-alter> alongside: it is what carries the pitch, and a reader that
    does not know the microtonal glyph names still transposes correctly.
    """
    key = mxl.XMLKey()
    attributes.add_child(key)
    for symbol in accidentals:
        alter = LIFT_TO_ALTER.get(symbol.lift)
        key.add_child(mxl.XMLKeyStep(value_=symbol.pitch[0]))
        key.add_child(mxl.XMLKeyAlter(value_=alter if alter is not None else 0))
        accidental = LIFT_TO_ACCIDENTAL.get(symbol.lift)
        if accidental is not None:
            key.add_child(mxl.XMLKeyAccidental(value_=accidental))


def get_staff(symbol: EncodedSymbol) -> int:
    return 2 if symbol.position == "lower" else 1


def get_xml_voice(staff_num: int, rhythmic_layer: int) -> int:
    """Build a stable MusicXML voice number per staff and rhythmic layer.

    Voice numbers are part-global in MusicXML, so using only the staff number can merge
    independent layers on the same staff. Reserve 4 voices per staff:
    staff 1 -> voices 1-4, staff 2 -> voices 5-8.
    """
    return (staff_num - 1) * 4 + rhythmic_layer + 1


@dataclass
class TimedNoteEvent:
    staff_num: int
    start: int
    end: int
    notes: list[mxl.XMLNote]


def rebalance_measure_voices(measure: mxl.XMLMeasure) -> None:
    """Assign stable non-overlapping voices per staff for a whole measure."""
    timed_events: list[TimedNoteEvent] = []
    current_time = 0
    last_note_start = 0
    for child in measure.get_children():
        if isinstance(child, mxl.XMLBackup):
            durations = child.get_children_of_type(mxl.XMLDuration)
            if len(durations) > 0:
                current_time -= int(durations[0].value_)
            continue
        if child.__class__.__name__ == "XMLForward":
            durations = child.get_children_of_type(mxl.XMLDuration)
            if len(durations) > 0:
                current_time += int(durations[0].value_)
            continue
        if not isinstance(child, mxl.XMLNote):
            continue

        duration_nodes = child.get_children_of_type(mxl.XMLDuration)
        duration = int(duration_nodes[0].value_) if len(duration_nodes) > 0 else 0
        staff_nodes = child.get_children_of_type(mxl.XMLStaff)
        staff_num = int(staff_nodes[0].value_) if len(staff_nodes) > 0 else 1
        is_chord_tone = len(child.get_children_of_type(mxl.XMLChord)) > 0
        start = last_note_start if is_chord_tone else current_time
        end = start + duration
        if is_chord_tone and (
            len(timed_events) > 0
            and timed_events[-1].staff_num == staff_num
            and timed_events[-1].start == start
            and timed_events[-1].end == end
        ):
            timed_events[-1].notes.append(child)
        elif is_chord_tone:
            timed_events.append(TimedNoteEvent(staff_num, start, end, [child]))
        else:
            last_note_start = start
            current_time += duration
            timed_events.append(TimedNoteEvent(staff_num, start, end, [child]))

    by_staff: dict[int, list[TimedNoteEvent]] = defaultdict(list)
    for event in timed_events:
        by_staff[event.staff_num].append(event)

    for staff_num, events in by_staff.items():
        sorted_events = sorted(events, key=lambda e: (e.start, e.end))
        active: list[tuple[int, int]] = []  # (end, local voice number 1..n)
        for event in sorted_events:
            active = [
                (active_end, voice_no)
                for active_end, voice_no in active
                if active_end > event.start
            ]
            used_voices = {voice_no for _, voice_no in active}
            voice_no = 1
            while voice_no in used_voices:
                voice_no += 1
            active.append((event.end, voice_no))
            xml_voice = str(get_xml_voice(staff_num, voice_no - 1))
            for note in event.notes:
                voice_nodes = note.get_children_of_type(mxl.XMLVoice)
                if len(voice_nodes) > 0:
                    voice_nodes[0].value_ = xml_voice


def build_clef(model_clef: EncodedSymbol, attributes: mxl.XMLAttributes) -> None:
    sign_and_line = model_clef.rhythm.split("_")[1]
    sign = sign_and_line[0]
    line = sign_and_line[1]
    clef = mxl.XMLClef(number=get_staff(model_clef))
    attributes.add_child(clef)
    clef.add_child(mxl.XMLSign(value_=sign))
    clef.add_child(mxl.XMLLine(value_=int(line)))


def build_time_signature(
    model_time_signature: EncodedSymbol, attributes: mxl.XMLAttributes, state: ConversionState
) -> None:
    time = mxl.XMLTime()

    head, denominator = model_time_signature.rhythm.split("/")
    attributes.add_child(time)
    # timeSignature_9/8 states both numbers; the older timeSignature/8 records
    # only the beat type, so the beat count has to be inferred from the measure
    # lengths. Turkish usuls (9/8, 10/8, 28/4) need the stated number.
    _, _, stated = head.partition("_")
    if stated.isdigit():
        beats = int(stated)
    else:
        beats = max(int(state.nominator * int(denominator)), 1)
    time.add_child(mxl.XMLBeats(value_=str(beats)))
    time.add_child(mxl.XMLBeatType(value_=denominator))
    state.beats = beats


def build_barline_style(barline: EncodedSymbol, xml: mxl.XMLBarline) -> None:
    style_value = "heavy-heavy" if barline.rhythm == "bolddoublebarline" else "light-light"
    style = mxl.XMLBarStyle(value_=style_value)
    xml.add_child(style)


def build_barline_ending(volta: EncodedSymbol, xml: mxl.XMLBarline, volta_number: int) -> None:
    if len(xml.get_children_of_type(mxl.XMLEnding)) > 0:
        eprint("barline already has an ending")
        return
    if volta.rhythm.startswith("voltaStart"):
        ending = mxl.XMLEnding(type="start", number=str(volta_number))
    elif volta.rhythm.startswith("voltaStop"):
        ending = mxl.XMLEnding(type="stop", number=str(volta_number))
    elif volta.rhythm.startswith("voltaDiscontinue"):
        ending = mxl.XMLEnding(type="discontinue", number=str(volta_number))
    else:
        raise ValueError("Unknown ending " + str(volta))
    xml.add_child(ending)


def build_repeat(barline: EncodedSymbol, xml: mxl.XMLBarline) -> None:
    if len(xml.get_children_of_type(mxl.XMLRepeat)) > 0:
        eprint("barline already has a repeat")
        return

    repeat = mxl.XMLRepeat()
    direction = "forward" if barline.rhythm == "repeatStart" else "backward"
    repeat._set_attributes({"direction": direction})
    xml.add_child(repeat)


LIFT_TO_ALTER = {
    "N": 0,
    "#": 1,
    "##": 2,
    "b": -1,
    "bb": -2,
}

# The makam lift tokens are named by comma count (see vocabulary.aeu_accidentals).
# A whole tone is 9 commas and 2 semitones, so one comma is 2/9 of a semitone.
# <alter> carries the sounding pitch; <accidental> below carries the printed glyph.
LIFT_TO_ALTER.update(
    {
        f"{'sharp' if c > 0 else 'flat'}{abs(c)}": round(c * 2 / 9, 3)
        for c in (1, 2, 3, 4, 5, 8, -1, -2, -3, -4, -5, -8)
    }
)

# Printed glyph. MusicXML names only six of the ten makam accidentals, so the
# 2- and 3-comma steps fall back to the nearest named glyph and lose a little
# information on export - the <alter> above stays exact either way. Change this
# table freely; it does not affect what the model learns.
LIFT_TO_ACCIDENTAL = {
    "sharp1": "quarter-sharp",
    "sharp2": "quarter-sharp",
    "sharp3": "sharp",
    "sharp4": "sharp",
    "sharp5": "slash-quarter-sharp",
    "sharp8": "slash-sharp",
    "flat1": "quarter-flat",
    "flat2": "quarter-flat",
    "flat3": "slash-flat",
    "flat4": "slash-flat",
    "flat5": "flat",
    "flat8": "double-slash-flat",
}

DURATION_NAMES = {
    0: "breve",
    1: "whole",
    2: "half",
    4: "quarter",
    8: "eighth",
    16: "16th",
    32: "32nd",
    64: "64th",
    128: "128th",
}


def build_articulations(
    note: mxl.XMLNote, articualations: str, tuplet_mark: str, state: ConversionState
) -> None:
    notation = mxl.XMLNotations()
    note.add_child(notation)

    xml_articulations = []
    xml_ornaments = []

    for articulation in articualations.split("_"):
        if articulation == "":
            continue
        elif articulation == nonote:
            eprint("WARNING note without valid articulation", articualations)
        elif articulation == "fermata":
            notation.add_child(mxl.XMLFermata())
        elif articulation == "arpeggiate":
            notation.add_child(mxl.XMLArpeggiate())
        elif articulation == "accent":
            xml_articulations.append(mxl.XMLAccent())
        elif articulation == "arpeggiate":
            notation.add_child(mxl.XMLArpeggiate())
        elif articulation == "fermata":
            xml_articulations.append(mxl.XMLFermata())
        elif articulation == "staccato":
            xml_articulations.append(mxl.XMLStaccato())
        elif articulation == "staccatissimo":
            xml_articulations.append(mxl.XMLStaccatissimo())
        elif articulation == "tenuto":
            xml_articulations.append(mxl.XMLTenuto())
        elif articulation == "tremolo":
            xml_ornaments.append(mxl.XMLTremolo(value_=3, type=state.toggle_tremolo_state()))
        elif articulation == "trill":
            xml_ornaments.append(mxl.XMLTrillMark())
        elif articulation == "breathMark":
            xml_articulations.append(mxl.XMLBreathMark())
        elif articulation == "turn":
            xml_ornaments.append(mxl.XMLInvertedTurn())
        elif articulation == "caesura":
            xml_articulations.append(mxl.XMLCaesura())
        elif articulation == "doit":
            xml_articulations.append(mxl.XMLDoit())
        elif articulation == "slurStart":
            notation.add_child(mxl.XMLSlur(type="start"))
        elif articulation == "slurStop":
            notation.add_child(mxl.XMLSlur(type="stop"))
        elif articulation == "tieStart":
            notation.add_child(mxl.XMLTied(type="start"))
        elif articulation == "tieStop":
            notation.add_child(mxl.XMLTied(type="stop"))
        else:
            raise ValueError("Unsupported articulation " + articulation)

    if tuplet_mark != "":
        tuplet_xml = mxl.XMLTuplet(type=tuplet_mark)
        notation.add_child(tuplet_xml)

    if len(xml_articulations) > 0:
        parent = mxl.XMLArticulations()
        for child in xml_articulations:
            parent.add_child(child)
        notation.add_child(parent)

    if len(xml_ornaments) > 0:
        parent = mxl.XMLOrnaments()
        for child in xml_ornaments:
            parent.add_child(child)
        notation.add_child(parent)


def build_note_or_rest(
    model_note: EncodedSymbol,
    rhythmic_layer: int,
    is_chord: bool,
    state: ConversionState,
    tuplet_mark: str,
) -> mxl.XMLNote:
    note = mxl.XMLNote()
    if is_chord:
        note.add_child(mxl.XMLChord())
    model_pitch = model_note.pitch
    model_duration = model_note.get_duration()
    if model_pitch == empty:
        if model_duration.fraction.numerator == 0:
            note.add_child(mxl.XMLRest(measure="yes"))
        else:
            note.add_child(mxl.XMLRest())
    elif model_pitch == nonote:
        eprint("WARNING note without pitch", model_note)
        note.add_child(mxl.XMLRest())
    else:
        pitch = mxl.XMLPitch()
        pitch.add_child(mxl.XMLStep(value_=model_pitch[0]))
        pitch.add_child(mxl.XMLOctave(value_=int(model_pitch[1])))
        if model_note.lift == nonote:
            eprint("WARNING note with invalid lift", model_note)
        else:
            # <alter> is the sounding pitch, and on a makam page most of it is
            # implied by the key signature and the measure rather than printed
            # on the note. homr.makam_key works that out; without it every note
            # the signature covers came out unaltered.
            sounding = model_note.sounding if model_note.sounding is not None else model_note.lift
            if sounding != empty:
                alter_value = LIFT_TO_ALTER.get(sounding)
                if alter_value is not None:
                    pitch.add_child(mxl.XMLAlter(value_=alter_value))
        note.add_child(pitch)
        # Emit an explicit <accidental> for the makam glyphs so the renderer
        # (e.g. MuseScore) draws the correct microtonal symbol.
        if model_note.lift in aeu_accidentals:
            note.add_child(mxl.XMLAccidental(value_=LIFT_TO_ACCIDENTAL[model_note.lift]))

    if "G" in model_note.rhythm:
        note.add_child(mxl.XMLGrace())
        base_duration = model_duration.kern
        duration_name = DURATION_NAMES[base_duration]
        note.add_child(mxl.XMLType(value_=duration_name))
    elif model_duration.fraction.numerator > 0:
        base_duration = 1 if model_duration.kern == 0 else model_duration.kern
        duration_name = DURATION_NAMES[base_duration]
        note.add_child(mxl.XMLType(value_=duration_name))
        note.add_child(mxl.XMLDuration(value_=int(model_duration.fraction * state.division)))
    else:
        duration_name = DURATION_NAMES[0]
        note.add_child(mxl.XMLType(value_=duration_name))
        note.add_child(mxl.XMLDuration(value_=state.beats))

    staff_num = get_staff(model_note)
    note.add_child(mxl.XMLStaff(value_=staff_num))
    note.add_child(mxl.XMLVoice(value_=str(get_xml_voice(staff_num, rhythmic_layer))))
    for _ in range(model_duration.dots):
        note.add_child(mxl.XMLDot())
    if model_duration.actual_notes != model_duration.normal_notes:
        time_modification = mxl.XMLTimeModification()
        time_modification.add_child(mxl.XMLActualNotes(value_=model_duration.actual_notes))
        time_modification.add_child(mxl.XMLNormalNotes(value_=model_duration.normal_notes))
        note.add_child(time_modification)
        build_articulations(note, model_note.articulation, tuplet_mark, state)
    else:
        build_articulations(note, model_note.articulation, "", state)

    return note


def build_multi_measure_rest(
    symbol: EncodedSymbol, attributes: mxl.XMLAttributes
) -> mxl.XMLMeasureStyle:
    other_styles = attributes.get_children_of_type(mxl.XMLMeasureStyle)
    if len(other_styles) > 0:
        eprint("Measure already has a multi rest")
        return
    duration = int(symbol.rhythm.split("_")[1].replace("m", ""))
    style = mxl.XMLMeasureStyle()
    rest = mxl.XMLMultipleRest(value_=duration)
    style.add_child(rest)
    attributes.add_child(style)


def build_note_chord(
    note_chord: SymbolChord, state: ConversionState, chord_duration: Fraction
) -> list[mxl.XMLElement]:
    # Ties and slurs used to be stripped here and dropped on the floor. That
    # matters for a makam score: a note too long for one symbol is engraved as
    # tied halves, so the file came out with two notes where the page has one --
    # 37 of them in a single sarki. Each note carries its own marking, which is
    # what MusicXML expects of a chord as well.
    by_duration = _group_notes(note_chord.symbols)
    result: list[mxl.XMLElement] = []
    final_duration = Fraction(0)
    sorted_durations = sorted(by_duration)
    for i, group_duration in enumerate(sorted_durations):
        is_first = True
        for note_loop in by_duration[group_duration]:
            note = note_loop
            result.append(build_note_or_rest(note, i, not is_first, state, note_chord.tuplet_mark))
            is_first = False
        if i != len(sorted_durations) - 1 and group_duration > Fraction(0):
            backup = mxl.XMLBackup()
            backup.add_child(mxl.XMLDuration(value_=int(group_duration * state.division)))
            result.append(backup)

        final_duration = group_duration

    # Reset the position to match the chord position
    if chord_duration < final_duration:
        backup = mxl.XMLBackup()
        backup.add_child(
            mxl.XMLDuration(value_=int((final_duration - chord_duration) * state.division))
        )
        result.append(backup)
    return result


def _group_notes(notes: list[EncodedSymbol]) -> dict[Fraction, list[EncodedSymbol]]:
    groups_by_duration = defaultdict(list)
    max_duration = max([n.get_duration().fraction for n in notes])
    for note in notes:
        duration = note.get_duration()
        is_grace = "G" in note.rhythm
        if is_grace:
            fraction = Fraction(0)
        elif duration.fraction.numerator == 0:
            # Whole measure rest
            fraction = max_duration
        else:
            fraction = duration.fraction
        groups_by_duration[fraction].append(note)
    return groups_by_duration


def build_add_time_direction(args: XmlGeneratorArguments) -> mxl.XMLDirection | None:
    if not args.metronome:
        return None
    direction = mxl.XMLDirection()
    direction_type = mxl.XMLDirectionType()
    direction.add_child(direction_type)
    metronome = mxl.XMLMetronome()
    direction_type.add_child(metronome)
    beat_unit = mxl.XMLBeatUnit(value_="quarter")
    metronome.add_child(beat_unit)
    per_minute = mxl.XMLPerMinute(value_=str(args.metronome))
    metronome.add_child(per_minute)
    if args.tempo:
        direction.add_child(mxl.XMLSound(tempo=args.tempo))
    else:
        direction.add_child(mxl.XMLSound(tempo=args.metronome))
    return direction


def find_common_division(durations: list[Fraction]) -> int:
    """
    Find the smallest division (denominator) so that all durations
    can be expressed as integer multiples.
    """

    def lcm(a: int, b: int) -> int:
        return abs(a * b) // math.gcd(a, b)

    denominators = [d.denominator for d in durations if d > 0]
    if not denominators:
        return 1
    common = denominators[0]
    for d in denominators[1:]:
        common = lcm(common, d)
    return common


def find_division_and_time_signature_nominator(voice: list[SymbolChord]) -> tuple[int, Fraction]:
    durations = [Fraction(1, 4)]
    duration_in_measure = Fraction(0)
    measure_duration = []
    for chord in voice:
        if chord.is_barline() and duration_in_measure > Fraction(0):
            measure_duration.append(duration_in_measure)
            duration_in_measure = Fraction(0)
        else:
            duration = chord.get_duration()
            if duration > Fraction(0):
                durations.append(duration)
                duration_in_measure += duration

    if duration_in_measure > Fraction(0):
        measure_duration.append(duration_in_measure)
        duration_in_measure = Fraction(0)

    if len(measure_duration) == 0:
        return find_common_division(durations), Fraction(1)

    nominator: Fraction = np.median(measure_duration)  # type: ignore

    return find_common_division(durations), nominator


def group_into_chords(voice: list[EncodedSymbol]) -> list[SymbolChord]:
    return [SymbolChord(s) for s in sort_token_chords(voice)]


def add_tuplet_start_stop(groups: list[SymbolChord]) -> list[SymbolChord]:
    has_tuplet_mark = False
    for group in groups:
        is_tuplet = False
        for symbol in group.symbols:
            if symbol.rhythm.startswith(("note", "rest")):
                duration = symbol.get_duration()
                is_tuplet = is_tuplet or duration.normal_notes != duration.actual_notes

        if is_tuplet != has_tuplet_mark:
            if has_tuplet_mark:
                group.tuplet_mark = "stop"
            else:
                group.tuplet_mark = "start"
            has_tuplet_mark = is_tuplet

    return groups


def build_divisions(division: int) -> mxl.XMLDivisions:
    # The divisions element indicates how many divisions per quarter(!) note are
    # used to indicate a note's duration
    # https://usermanuals.musicxml.com/MusicXML/Content/EL-MusicXML-divisions.htm
    return mxl.XMLDivisions(value_=division // 4)


if __name__ == "__main__":
    import sys

    from training.transformer.training_vocabulary import read_tokens

    file = "tabi_measure.tokens"
    if len(sys.argv) > 1:
        file = sys.argv[1]
    tokens = read_tokens(file)
    xml = generate_xml(XmlGeneratorArguments(True), [tokens], "")
    xml.write(file.replace(".tokens", ".musicxml"))
