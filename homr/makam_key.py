"""Turn what the page draws into what the music sounds.

The model reports engraving: the symbols actually printed on the staff. Most
altered notes carry no symbol at all, because the makam key signature already
covers them, or because the same note was altered earlier in the measure. Four
of every five alterations on a real Mus2 page are implied this way.

MusicXML wants the other thing. Its <alter> is the sounding pitch, and its
<accidental> is the glyph. So between reading the page and writing the file, the
implied alterations have to be put back:

    keyAccidental B4 flat1     the signature says every B is a comma flat
    note_8 B4 _                nothing drawn, so B4 sounds a comma flat
    note_8 B4 N                a natural is drawn, so B4 sounds natural
    note_8 B4 _                still natural: the measure remembers
    barline                    the memory is cleared
    note_8 B4 _                a comma flat again, from the signature

This is the exact inverse of what the training data generator does when it works
out which symbols a page would print. It is not quite lossless: a work that
engraves the 2-comma flat with the 1-comma sign gives no way to tell the two
apart afterwards, so the rounding a score applies cannot be undone.
"""

from homr.transformer.vocabulary import (
    EncodedSymbol,
    aeu_commas,
    empty,
    key_accidental,
    nonote,
)

MEASURE_ENDS = ("barline", "repeat", "volta", "bolddoublebarline")


def commas_for_lift(lift: str) -> int | None:
    """How many commas a drawn symbol raises or lowers by, or None if it is not one."""
    if lift == "N":
        return 0
    for prefix, sign in (("sharp", 1), ("flat", -1)):
        if lift.startswith(prefix):
            value = lift[len(prefix) :]
            if value.isdigit() and int(value) in aeu_commas:
                return sign * int(value)
    return None


def lift_for_commas(commas: int) -> str:
    if commas == 0:
        return "N"
    return f"{'sharp' if commas > 0 else 'flat'}{abs(commas)}"


class MakamKey:
    """The key signature plus what the current measure has already been told.

    The signature is held per letter, because it covers every octave; the
    measure's own accidentals are held per letter and octave, because they do
    not carry from one octave to another.
    """

    def __init__(self) -> None:
        self.signature: dict[str, int] = {}
        self.measure: dict[str, int] = {}
        # A note too long for one symbol is engraved as tied halves, and the tie
        # can cross a barline. The far half carries no sign of its own and the
        # measure it lands in has forgotten the near half, so it has to be told.
        self.tied: dict[str, int] = {}

    def read_signature(self, pitch: str, lift: str) -> None:
        commas = commas_for_lift(lift)
        if commas is not None and pitch:
            self.signature[pitch[0]] = commas

    def start_measure(self) -> None:
        self.measure.clear()

    def carry_tie(self, pitch: str, commas: int) -> None:
        self.tied[pitch] = commas

    def take_tie(self, pitch: str) -> int | None:
        return self.tied.get(pitch)

    def sounding(self, pitch: str, drawn: str) -> int:
        commas = commas_for_lift(drawn)
        if commas is not None:
            self.measure[pitch] = commas
            return commas
        if pitch in self.measure:
            return self.measure[pitch]
        return self.signature.get(pitch[0], 0)


def _ends_measure(rhythm: str) -> bool:
    return any(rhythm.startswith(mark) for mark in MEASURE_ENDS)


def resolve_sounding(symbols: list[EncodedSymbol]) -> list[EncodedSymbol]:
    """Give every note the alteration it sounds, leaving the drawn one alone.

    The signature is rebuilt from the keyAccidental symbols as they arrive, so a
    staff that restates it -- Mus2 restates it on every system -- simply
    confirms what is already known.
    """
    key = MakamKey()
    resolved = []
    for symbol in symbols:
        if symbol.rhythm == key_accidental:
            key.read_signature(symbol.pitch, symbol.lift)
        elif _ends_measure(symbol.rhythm):
            key.start_measure()
        elif symbol.pitch not in (nonote, empty) and symbol.lift != nonote:
            held = key.take_tie(symbol.pitch) if "tieStop" in symbol.articulation else None
            commas = held if held is not None else key.sounding(symbol.pitch, symbol.lift)
            if "tieStart" in symbol.articulation or "tieStop" in symbol.articulation:
                key.carry_tie(symbol.pitch, commas)
            # A plain note in a letter the signature never touches needs no
            # <alter> at all. One the signature does touch needs an explicit
            # zero, or a reader applying the signature would alter it anyway.
            if commas or symbol.lift == "N" or symbol.pitch[0] in key.signature:
                symbol = symbol.with_sounding(lift_for_commas(commas))
        resolved.append(symbol)
    return resolved
