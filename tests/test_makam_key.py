import unittest

from homr.makam_key import commas_for_lift, lift_for_commas, resolve_sounding
from homr.transformer.vocabulary import EncodedSymbol


def note(rhythm: str, pitch: str, lift: str, articulation: str = "_") -> EncodedSymbol:
    return EncodedSymbol(rhythm, pitch, lift, articulation, "upper")


BARLINE = EncodedSymbol("barline", ".", ".", ".", ".")


class TestMakamKey(unittest.TestCase):

    def test_commas_round_trip(self) -> None:
        for commas in (-8, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 8):
            self.assertEqual(commas_for_lift(lift_for_commas(commas)), commas)

    def test_western_and_blank_lifts_are_not_makam_accidentals(self) -> None:
        for lift in ("_", ".", "#", "b", "##", "bb", "sharp7", "flat9"):
            self.assertIsNone(commas_for_lift(lift))

    def test_key_signature_reaches_a_note_that_carries_no_sign(self) -> None:
        symbols = [
            EncodedSymbol("keyAccidental", "B4", "flat1", "_", "upper"),
            EncodedSymbol("keyAccidental", "F5", "sharp4", "_", "upper"),
            note("note_8", "F5", "_"),
            note("note_4", "C5", "_"),
        ]
        resolved = resolve_sounding(symbols)
        self.assertEqual(resolved[2].sounding, "sharp4")
        # C is not in the signature, so nothing has to be written for it.
        self.assertIsNone(resolved[3].sounding)

    def test_the_signature_covers_every_octave(self) -> None:
        symbols = [
            EncodedSymbol("keyAccidental", "B4", "flat1", "_", "upper"),
            note("note_8", "B4", "_"),
            note("note_8", "B5", "_"),
        ]
        resolved = resolve_sounding(symbols)
        self.assertEqual(resolved[1].sounding, "flat1")
        self.assertEqual(resolved[2].sounding, "flat1")

    def test_a_sign_holds_to_the_barline_and_only_in_its_own_octave(self) -> None:
        symbols = [
            EncodedSymbol("keyAccidental", "B4", "flat1", "_", "upper"),
            note("note_8", "B4", "N"),
            note("note_8", "B4", "_"),
            note("note_8", "B5", "_"),
            BARLINE,
            note("note_8", "B4", "_"),
        ]
        resolved = resolve_sounding(symbols)
        self.assertEqual(resolved[1].sounding, "N")
        # Still natural: the measure remembers.
        self.assertEqual(resolved[2].sounding, "N")
        # The octave above was never told, so the signature still applies.
        self.assertEqual(resolved[3].sounding, "flat1")
        # The barline clears the memory.
        self.assertEqual(resolved[5].sounding, "flat1")

    def test_a_repeat_sign_ends_the_measure_too(self) -> None:
        symbols = [
            EncodedSymbol("keyAccidental", "B4", "flat1", "_", "upper"),
            note("note_8", "B4", "N"),
            EncodedSymbol("repeatEnd", ".", ".", ".", "."),
            note("note_8", "B4", "_"),
        ]
        resolved = resolve_sounding(symbols)
        self.assertEqual(resolved[3].sounding, "flat1")

    def test_a_tie_carries_its_sign_over_the_barline(self) -> None:
        symbols = [
            EncodedSymbol("keyAccidental", "F5", "sharp4", "_", "upper"),
            note("note_2", "F5", "sharp5", "tieStart"),
            BARLINE,
            note("note_2", "F5", "_", "tieStop"),
            note("note_4", "F5", "_"),
        ]
        resolved = resolve_sounding(symbols)
        self.assertEqual(resolved[1].sounding, "sharp5")
        # The far half of the tie is the same note, however the measure was reset.
        self.assertEqual(resolved[3].sounding, "sharp5")
        # A fresh note after it goes back to the signature.
        self.assertEqual(resolved[4].sounding, "sharp4")

    def test_a_restated_signature_changes_nothing(self) -> None:
        """Mus2 redraws the key signature at the head of every system."""
        symbols = [
            EncodedSymbol("keyAccidental", "B4", "flat1", "_", "upper"),
            note("note_8", "B4", "N"),
            EncodedSymbol("keyAccidental", "B4", "flat1", "_", "upper"),
            note("note_8", "B4", "_"),
        ]
        resolved = resolve_sounding(symbols)
        self.assertEqual(resolved[3].sounding, "N")

    def test_rests_and_barlines_are_left_alone(self) -> None:
        symbols = [
            EncodedSymbol("keyAccidental", "B4", "flat1", "_", "upper"),
            note("rest_4", "_", "_"),
            BARLINE,
        ]
        resolved = resolve_sounding(symbols)
        self.assertIsNone(resolved[1].sounding)
        self.assertIsNone(resolved[2].sounding)


if __name__ == "__main__":
    unittest.main()
