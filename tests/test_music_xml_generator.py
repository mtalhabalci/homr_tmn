# ruff: noqa: E501

import re
import unittest
import xml.etree.ElementTree as ET
from typing import Any

import musicxml.xmlelement.xmlelement as mxl

from homr.music_xml_generator import (
    SymbolChord,
    XmlGeneratorArguments,
    generate_xml,
    rebalance_measure_voices,
    slurs_on_equal_notes_as_ties,
)
from homr.transformer.vocabulary import EncodedSymbol
from training.transformer.training_vocabulary import (
    read_token_lines,
)


class TestMusicXmlGenerator(unittest.TestCase):
    """
    MusicXML testing is mostly covered by training/validate_music_xml_conversion.py
    This script requires that the data sets are downloaded and converted and uses
    the data sets to check that back and forth conversion works.
    """

    def test_chord_with_different_duratons(self) -> None:
        tabi_measure_18_upper = """clef_G2 . . . upper
keySignature_4 . . . .
timeSignature/8 . . . .
note_4. G3 # _ upper &note_4. C4 # _ upper&note_16 E4 # _ upper
note_16 F4 # _ upper
note_4 E4 # _ upper
note_8 E4 # _ upper
note_8 C4 # _ upper
note_8 D4 # _ upper
barline . . . ."""
        tokens = read_token_lines(tabi_measure_18_upper.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")
        actual = self._xml_to_str(xml)
        expected = """XMLScorePartwise([XMLWork([XMLWorkTitle()]),
XMLPart([XMLMeasure([XMLAttributes([XMLDivisions(value: 4)]),
XMLAttributes([XMLKey([XMLFifths(value: 4)]),
XMLTime([XMLBeats(value: 6),
XMLBeatType(value: 8)]),
XMLClef([XMLSign(value: G),
XMLLine(value: 2)])]),
XMLNote([XMLPitch([XMLStep(value: E)]),
XMLDuration(value: 1),
XMLVoice(value: 1),
XMLStaff(value: 1),
XMLNotations()]),
XMLBackup([XMLDuration(value: 1)]),
XMLNote([XMLPitch([XMLStep(value: G)]),
XMLDuration(value: 6),
XMLVoice(value: 2),
XMLDot(),
XMLStaff(value: 1),
XMLNotations()]),
XMLNote([XMLChord(),
XMLPitch([XMLStep(value: C)]),
XMLDuration(value: 6),
XMLVoice(value: 2),
XMLDot(),
XMLStaff(value: 1),
XMLNotations()]),
XMLBackup([XMLDuration(value: 5)]),
XMLNote([XMLPitch([XMLStep(value: F)]),
XMLDuration(value: 1),
XMLVoice(value: 1),
XMLStaff(value: 1),
XMLNotations()]),
XMLNote([XMLPitch([XMLStep(value: E)]),
XMLDuration(value: 4),
XMLVoice(value: 1),
XMLStaff(value: 1),
XMLNotations()]),
XMLNote([XMLPitch([XMLStep(value: E)]),
XMLDuration(value: 2),
XMLVoice(value: 1),
XMLStaff(value: 1),
XMLNotations()]),
XMLNote([XMLPitch([XMLStep(value: C)]),
XMLDuration(value: 2),
XMLVoice(value: 1),
XMLStaff(value: 1),
XMLNotations()]),
XMLNote([XMLPitch([XMLStep(value: D)]),
XMLDuration(value: 2),
XMLVoice(value: 1),
XMLStaff(value: 1),
XMLNotations()])])])])"""
        self.assertEqual(self._norm_expected(expected), actual)

    def test_grand_staff_generation(self) -> None:
        grandstaff = """clef_G2 _ _ _ upper&clef_F4 _ _ _ lower
keySignature_1 . . . .
timeSignature/4 . . . .
note_1 G4 _ _ upper&note_1 A3 # _ upper&rest_2 _ _ _ upper&note_4 G3 _ _ lower
rest_4 _ _ _ lower
note_2 E4 _ _ upper&note_2 C2 _ _ lower
barline . . . ."""
        tokens = read_token_lines(grandstaff.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")
        actual = self._xml_to_str(xml)
        expected = """XMLScorePartwise([XMLWork([XMLWorkTitle()]),
XMLPart([XMLMeasure([XMLAttributes([XMLDivisions(value: 1)]),
XMLAttributes([XMLKey([XMLFifths(value: 1)]),
XMLTime([XMLBeats(value: 4),
XMLBeatType(value: 4)]),
XMLClef([XMLSign(value: G),
XMLLine(value: 2)]),
XMLClef([XMLSign(value: F),
XMLLine(value: 4)])]),
XMLNote([XMLRest(),
XMLDuration(value: 2),
XMLVoice(value: 1),
XMLStaff(value: 1),
XMLNotations()]),
XMLBackup([XMLDuration(value: 2)]),
XMLNote([XMLPitch([XMLStep(value: G)]),
XMLDuration(value: 4),
XMLVoice(value: 2),
XMLStaff(value: 1),
XMLNotations()]),
XMLNote([XMLChord(),
XMLPitch([XMLStep(value: A)]),
XMLDuration(value: 4),
XMLVoice(value: 2),
XMLStaff(value: 1),
XMLNotations()]),
XMLBackup([XMLDuration(value: 4)]),
XMLNote([XMLPitch([XMLStep(value: G)]),
XMLDuration(value: 1),
XMLVoice(value: 5),
XMLStaff(value: 2),
XMLNotations()]),
XMLNote([XMLRest(),
XMLDuration(value: 1),
XMLVoice(value: 5),
XMLStaff(value: 2),
XMLNotations()]),
XMLNote([XMLPitch([XMLStep(value: E)]),
XMLDuration(value: 2),
XMLVoice(value: 1),
XMLStaff(value: 1),
XMLNotations()]),
XMLBackup([XMLDuration(value: 2)]),
XMLNote([XMLPitch([XMLStep(value: C)]),
XMLDuration(value: 2),
XMLVoice(value: 5),
XMLStaff(value: 2),
XMLNotations()])])])])"""
        self.assertEqual(self._norm_expected(expected), actual)

    def test_a_tie_reaches_the_file(self) -> None:
        """A note too long for one symbol is engraved as tied halves.

        The two halves used to arrive as two separate notes, because ties were
        stripped off before the note was written: the file then said two notes
        where the page shows one held note.
        """
        held_note = """clef_G2 . . . upper
note_2 E5 _ tieStart upper
barline . . . .
note_8 E5 _ tieStop upper
barline . . . ."""
        tokens = read_token_lines(held_note.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "").to_string()
        self.assertIn('<tied type="start"', xml)
        self.assertIn('<tied type="stop"', xml)

    def test_a_slur_over_two_equal_notes_is_a_tie(self) -> None:
        tokens = read_token_lines(
            """note_4 A4 _ slurStart upper
barline . . . .
note_8 A4 _ slurStop upper
note_8 B4 _ slurStart upper
note_8 C5 _ slurStop upper""".splitlines()
        )
        result = slurs_on_equal_notes_as_ties(tokens)
        self.assertEqual([s.articulation for s in result if s.rhythm.startswith("note")],
                         ["tieStart", "tieStop", "slurStart", "slurStop"])

    def test_a_slur_over_dotted_equal_notes_stays_a_slur(self) -> None:
        """Portato: two B4s under one slur, each with a staccato dot (Lieder lc5667843)."""
        tokens = read_token_lines(
            """note_8 B4 _ slurStart_staccato upper
note_8 B4 _ slurStop_staccato upper""".splitlines()
        )
        result = slurs_on_equal_notes_as_ties(tokens)
        self.assertEqual([s.articulation for s in result], ["slurStart_staccato", "slurStop_staccato"])

    def test_a_slur_to_a_note_with_its_own_sign_stays_a_slur(self) -> None:
        tokens = read_token_lines(
            """note_4 F5 sharp4 slurStart upper
note_4 F5 N slurStop upper""".splitlines()
        )
        result = slurs_on_equal_notes_as_ties(tokens)
        self.assertEqual([s.articulation for s in result], ["slurStart", "slurStop"])

    def test_a_measure_repeat_plays_the_measure_before(self) -> None:
        tokens = read_token_lines(
            """clef_G2 . . . upper
timeSignature_2/4 . . . .
note_4 A4 _ _ upper
note_4 C5 _ _ upper
barline . . . .
measureRepeat . . . .
barline . . . .""".splitlines()
        )
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "").to_string()
        second = xml.split('<measure number="2"')[1]
        self.assertEqual(second.count("<note"), 2)
        self.assertIn("<step>C</step>", second)
        # the notes are there for playback; the measure-style makes a reader draw the % sign
        self.assertIn('<measure-repeat type="start">1</measure-repeat>', second)

    def test_signs_are_written_by_their_musicxml_names(self) -> None:
        """MusicXML 4.0 names the THM numbered signs (flat-2 ...) and the AEU slash- signs."""
        tokens = read_token_lines(
            """clef_G2 . . . upper
keyAccidental B4 flat2 _ upper
timeSignature_2/4 . . . .
note_4 F5 N tieStart upper
note_4 F5 _ tieStop upper
barline . . . .
note_4 G5 ## _ upper
note_4 E5 sharp3 _ upper
barline . . . .""".splitlines()
        )
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "").to_string()
        self.assertIn('<key-accidental smufl="accidental2CommaFlat">flat-2</key-accidental>', xml)
        # each sign names its SMuFL glyph too, the THM numbered ones from the Turkish folk music range
        self.assertIn('<accidental smufl="accidentalNatural">natural</accidental>', xml)
        self.assertIn('<accidental smufl="accidentalDoubleSharp">double-sharp</accidental>', xml)
        self.assertIn('<accidental smufl="accidental3CommaSharp">sharp-3</accidental>', xml)
        # a tie both drawn and sounding
        self.assertIn('<tie type="start"', xml)
        self.assertIn('<tie type="stop"', xml)

    def test_aeu_signs_carry_their_smufl_names(self) -> None:
        """A reader that resolves accidentals by glyph needs the SMuFL name: a plain flat is 5 commas in AEU."""
        tokens = read_token_lines(
            """clef_G2 . . . upper
keyAccidental B4 flat1 _ upper
timeSignature_2/4 . . . .
note_4 C5 sharp4 _ upper
note_4 F5 sharp5 _ upper
barline . . . .
note_4 E5 flat2 _ upper
note_4 G5 sharp3 _ upper
barline . . . .""".splitlines()
        )
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "").to_string()
        self.assertIn('<key-accidental smufl="accidentalKomaFlat">quarter-flat</key-accidental>', xml)
        self.assertIn('<accidental smufl="accidentalBakiyeSharp">sharp</accidental>', xml)
        self.assertIn('<accidental smufl="accidentalKucukMucennebSharp">slash-quarter-sharp</accidental>', xml)
        self.assertIn('<accidental smufl="accidental2CommaFlat">flat-2</accidental>', xml)
        self.assertIn('<accidental smufl="accidental3CommaSharp">sharp-3</accidental>', xml)

    def test_software_name_and_measure_confidence(self) -> None:
        """An application names itself in <software>; the lowest confidence of each measure is collected."""
        tokens = read_token_lines(
            """clef_G2 . . . upper
timeSignature_2/4 . . . .
note_4 C5 _ _ upper
note_4 D5 _ _ upper
barline . . . .
note_2 E5 _ _ upper
barline . . . .""".splitlines()
        )
        for symbol, sure in zip(tokens, (0.99, 0.98, 0.9, 0.4, 0.97, 0.8, 0.95), strict=False):
            symbol.confidence = sure
        confidence: dict[int, float] = {}
        xml = generate_xml(XmlGeneratorArguments(software="Uygulama", measure_confidence=confidence), [tokens],
                           "").to_string()
        self.assertIn("<software>Uygulama</software>", xml)
        self.assertNotIn("homr", xml)
        self.assertEqual(confidence, {1: 0.4, 2: 0.8})

    def test_repeats_and_endings_without_empty_measures(self) -> None:
        """A forward repeat opens its measure on the left, the second ending is numbered 2, and a repeat or a
        closing sign that follows a barline makes no measure of its own."""
        tokens = read_token_lines(
            """clef_G2 . . . upper
timeSignature_2/4 . . . .
repeatStart . . . .
note_2 C5 _ _ upper
barline . . . .
voltaStart . . . .
note_2 D5 _ _ upper
repeatEnd . . . .
voltaStop . . . .
voltaStart . . . .
note_2 E5 _ _ upper
barline . . . .
voltaStop . . . .
newline . . . .
repeatStart . . . .
note_2 F5 _ _ upper
bolddoublebarline . . . .
coda . . . .""".splitlines()
        )
        root = ET.fromstring(generate_xml(XmlGeneratorArguments(), [tokens], "").to_string())
        measures = root.findall("./part/measure")
        self.assertEqual([m.get("number") for m in measures], ["1", "2", "3", "4"])
        self.assertTrue(all(m.find("note") is not None for m in measures))
        self.assertEqual(measures[0].find("barline").get("location"), "left")
        self.assertEqual(measures[0].find("barline/repeat").get("direction"), "forward")
        self.assertEqual(measures[3].find("barline[@location='left']/repeat").get("direction"), "forward")
        endings = [(m.get("number"), b.get("location"), e.get("type"), e.get("number"))
                   for m in measures for b in m.findall("barline") for e in b.findall("ending")]
        self.assertEqual(endings, [("2", "left", "start", "1"), ("2", "right", "stop", "1"),
                                   ("3", "left", "start", "2"), ("3", "right", "stop", "2")])
        self.assertIsNotNone(measures[3].find("direction/direction-type/coda"))
        last = [c.tag for c in measures[3]]
        self.assertLess(last.index("direction"), len(last) - 1 - last[::-1].index("barline"))
        self.assertTrue(all(len(b) for m in measures for b in m.findall("barline")))

    def test_written_da_capo_and_dal_segno_carry_the_jump(self) -> None:
        """A written D.S. jumps back to the segno before it and D.C. to the start; a second segno stays a sign."""
        tokens = read_token_lines(
            """clef_G2 . . . upper
timeSignature_2/4 . . . .
note_2 C5 _ _ upper
barline . . . .
segno . . . .
note_2 D5 _ _ upper
barline . . . .
note_2 E5 _ _ upper
daSegno . . . .
barline . . . .
note_2 F5 _ _ upper
segno . . . .
daCapo . . . .
barline . . . .""".splitlines()
        )
        root = ET.fromstring(generate_xml(XmlGeneratorArguments(), [tokens], "").to_string())
        sounds = [(m.get("number"), d.find("sound").attrib if d.find("sound") is not None else None)
                  for m in root.findall("./part/measure") for d in m.findall("direction")]
        self.assertEqual(sounds, [("2", {"segno": "segno1"}), ("3", {"dalsegno": "segno1"}),
                                  ("4", {"segno": "segno2"}), ("4", {"dacapo": "yes"})])

    def test_staff_measures_give_each_staffs_measure_range(self) -> None:
        """Each staff's first and last measure; a measure that runs on past the line end belongs to both staffs."""
        tokens = read_token_lines(
            """clef_G2 . . . upper
timeSignature_2/4 . . . .
note_2 C5 _ _ upper
barline . . . .
note_4 D5 _ _ upper
newline . . . .
clef_G2 . . . upper
note_4 E5 _ _ upper
barline . . . .
note_2 F5 _ _ upper
barline . . . .
newline . . . .""".splitlines()
        )
        ranges: list[tuple[int, int]] = []
        generate_xml(XmlGeneratorArguments(staff_measures=ranges), [tokens], "")
        self.assertEqual(ranges, [(1, 2), (2, 3)])

    def test_tempo_marks_become_metronome_and_playback_tempo(self) -> None:
        """A tempo read off the page opens its measure: the printed mark and the tempo in quarters per minute."""
        tokens = read_token_lines(
            """clef_G2 . . . upper
timeSignature_2/4 . . . .
note_2 C5 _ _ upper
barline . . . .
note_2 D5 _ _ upper
barline . . . .""".splitlines()
        )
        args = XmlGeneratorArguments(tempo_marks={1: ("eighth", False, 168), 2: ("quarter", True, 60)})
        root = ET.fromstring(generate_xml(args, [tokens], "").to_string())
        marks = [(m.get("number"), d.findtext("direction-type/metronome/beat-unit"),
                  d.find("direction-type/metronome/beat-unit-dot") is not None,
                  d.findtext("direction-type/metronome/per-minute"), d.find("sound").get("tempo"))
                 for m in root.findall("./part/measure") for d in m.findall("direction")]
        self.assertEqual(marks, [("1", "eighth", False, "168", "84"), ("2", "quarter", True, "60", "90")])

    def test_first_ending_stops_at_its_backward_repeat(self) -> None:
        """When the end of a first ending's bracket was not read, the ending still stops at the backward repeat."""
        tokens = read_token_lines(
            """clef_G2 . . . upper
timeSignature_2/4 . . . .
note_2 C5 _ _ upper
barline . . . .
voltaStart . . . .
note_2 D5 _ _ upper
repeatEnd . . . .
voltaStart . . . .
note_2 E5 _ _ upper
barline . . . .""".splitlines()
        )
        root = ET.fromstring(generate_xml(XmlGeneratorArguments(), [tokens], "").to_string())
        endings = [(m.get("number"), b.get("location"), e.get("type"), e.get("number"))
                   for m in root.findall("./part/measure") for b in m.findall("barline") for e in b.findall("ending")]
        self.assertEqual(endings, [("2", "left", "start", "1"), ("2", "right", "stop", "1"),
                                   ("3", "left", "start", "2"), ("3", "right", "discontinue", "2")])

    def test_begin_chord_with_standalone_rests(self) -> None:
        """
        If the lower position consists of a standalone rest then start the
        chord with this. That fixes an issue where the upper position
        consists of tuplets because in that case backups must not be used.

        See tabi.jpg measure 9 for an example.
        """
        chord = SymbolChord(
            [
                EncodedSymbol("note_12", position="upper"),
                EncodedSymbol("note_12", position="upper"),
                EncodedSymbol("rest_8", position="lower"),
            ]
        )
        first, second = chord.into_positions()

        self.assertEqual(first.symbols, [EncodedSymbol("rest_8", position="lower")])
        self.assertEqual(
            second.symbols,
            [
                EncodedSymbol("note_12", position="upper"),
                EncodedSymbol("note_12", position="upper"),
            ],
        )

    def test_rebalance_measure_voices_assigns_stable_voices_per_staff(self) -> None:
        measure = mxl.XMLMeasure()

        note1 = self._build_test_note(duration=4, staff=1, voice=1)
        measure.add_child(note1)
        measure.add_child(self._build_test_backup(duration=4))

        # Overlaps note1, so this must move to voice 2.
        note2 = self._build_test_note(duration=2, staff=1, voice=1)
        measure.add_child(note2)

        # Starts later but still overlaps with note1, reusing voice 2.
        note3 = self._build_test_note(duration=2, staff=1, voice=1)
        measure.add_child(note3)

        # Chord tone sharing note3's timing; should get the same voice as note3.
        note4 = self._build_test_note(duration=2, staff=1, voice=1, is_chord=True)
        measure.add_child(note4)

        measure.add_child(self._build_test_backup(duration=4))
        note5 = self._build_test_note(duration=4, staff=2, voice=1)
        measure.add_child(note5)
        measure.add_child(self._build_test_backup(duration=4))

        # Overlaps note5 on staff 2.
        note6 = self._build_test_note(duration=2, staff=2, voice=1)
        measure.add_child(note6)

        rebalance_measure_voices(measure)

        # Same-start events are processed by (start, end), so shorter durations
        # receive lower local voice numbers first.
        self.assertEqual(self._read_note_voice(note1), "2")
        self.assertEqual(self._read_note_voice(note2), "1")
        self.assertEqual(self._read_note_voice(note3), "1")
        self.assertEqual(self._read_note_voice(note4), "1")
        self.assertEqual(self._read_note_voice(note5), "6")
        self.assertEqual(self._read_note_voice(note6), "5")

    def _norm_expected(self, expected: str) -> str:
        norm = expected.replace("\n", "")
        norm = re.sub(r",\s+", ",", norm)
        norm = re.sub(r"\[\s+", "[", norm)
        return norm

    def _xml_to_str(self, xml: Any) -> str:
        def recurse(node_or_list: Any) -> str:
            if isinstance(node_or_list, list):
                return (
                    "["
                    + ",".join(recurse(child) for child in node_or_list if child is not None)
                    + "]"
                )

            node = node_or_list
            name = node.__class__.__name__

            ignore_nodes = (
                "XMLAlter",
                "XMLOctave",
                "XMLType",
                "XMLPartList",
                "XMLDefaults",
                "XMLIdentification",
                "XMLEncoding",
                "XMLSoftware",
                "XMLStaves",
                "XMLPartSymbol",
            )

            if name in ignore_nodes:
                return ""
            value = getattr(node, "value_", None)

            if hasattr(node, "children"):
                children = node.children
            elif hasattr(node, "get_children"):
                children = node.get_children()
            else:
                children = []

            child_strs = [recurse(child) for child in children if child is not None]
            child_strs = [child for child in child_strs if child != ""]

            parts = []
            if value is not None and value != "":
                parts.append(f"value: {value}")
            if child_strs:
                parts.append(f"[{','.join(child_strs)}]")

            if parts:
                return f"{name}({','.join(parts)})"
            else:
                return f"{name}()"

        return recurse(xml)

    def _build_test_note(
        self, duration: int, staff: int, voice: int, is_chord: bool = False
    ) -> mxl.XMLNote:
        note = mxl.XMLNote()
        if is_chord:
            note.add_child(mxl.XMLChord())
        note.add_child(mxl.XMLDuration(value_=duration))
        note.add_child(mxl.XMLStaff(value_=staff))
        note.add_child(mxl.XMLVoice(value_=str(voice)))
        return note

    def _build_test_backup(self, duration: int) -> mxl.XMLBackup:
        backup = mxl.XMLBackup()
        backup.add_child(mxl.XMLDuration(value_=duration))
        return backup

    def _read_note_voice(self, note: mxl.XMLNote) -> str:
        voices = note.get_children_of_type(mxl.XMLVoice)
        self.assertGreater(len(voices), 0)
        return str(voices[0].value_)
