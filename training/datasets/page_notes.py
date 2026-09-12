"""Read the notes of a staff straight off the pdf, and line them up with labels.

The labels are worked out from the .mu2, but the pdf was not always engraved
from exactly that .mu2: some pages carry a courtesy sign the .mu2 knows nothing
about, some leave out one it has, and when barlines are miscounted a staff's
last measure can end up filed under the next staff. The page is what the model
sees, so the page is what the labels must describe.

Mus2 draws notation as text. Every notehead and accidental is a character whose
origin sits on its staff degree, so a notehead's pitch can be read from its
height alone, and the sign in front of it from the character just to its left.
"""

import fitz

from training.datasets.convert_symbtr import ACCIDENTAL_GLYPHS, NOTEHEAD_GLYPHS, _key_signature_glyphs
from training.datasets.shift_staff import MARGIN, _characters, _degree

NATURAL = 0x6E
LETTERS = "CDEFGAB"

# How far left of its notehead an accidental may start, in staff steps.
REACH = 5
# How far above or below the staff a notehead may sit, in staff steps.
LEDGER_REACH = 9


def lift_of_glyph(code: int) -> str | None:
    if code == NATURAL:
        return "N"
    commas = ACCIDENTAL_GLYPHS.get(code)
    if commas is None:
        return None
    return f"sharp{commas}" if commas > 0 else f"flat{-commas}"


def pitch_at(y: float, bottom: float, step: float) -> str:
    degree = round((bottom - y) / step) + _degree("E4")
    return f"{LETTERS[degree % 7]}{degree // 7}"


def uses_usual_encoding(document: "fitz.Document") -> bool:
    """False for the pdfs whose Mus2 characters do not follow the usual codes.

    About one work in eight embeds Mus2 as a CID font or otherwise renumbers
    its characters: the 2-comma sharp arrives as 0xC8, the 2-comma flat as the
    code that is the treble clef everywhere else. Noteheads still read right
    there, but accidentals cannot be read by code, so the page is not trusted
    for them.
    """
    for page in document:
        for font in page.get_fonts(full=True):
            if "Mus2" in font[3] and "Identity" in font[3]:
                return False
    return _key_signature_glyphs(document) is not None


def read_staff(page: "fitz.Page", staff: list[float]) -> list[dict]:
    """The noteheads of one staff, left to right, each with the sign before it."""
    top, bottom = staff[0], staff[-1]
    step = (bottom - top) / 8
    band = fitz.Rect(0, top - MARGIN, page.rect.width, bottom + MARGIN)
    glyphs = [g for g in _characters(page, band) if g["notation"]]
    heads = []
    for glyph in glyphs:
        if glyph["code"] not in NOTEHEAD_GLYPHS:
            continue
        y = glyph["origin"][1]
        # Further out than four ledger lines is not this staff's: the band also
        # catches the small notes some scores print beneath it for the usul.
        if not top - LEDGER_REACH * step <= y <= bottom + LEDGER_REACH * step:
            continue
        # Mus2 builds some notes from two glyphs laid on the same spot -- a head
        # and the flag or stem that goes with it. One note, not two.
        if heads and abs(heads[-1]["x"] - glyph["x"]) < 1 and abs(heads[-1]["origin"][1] - y) < 1:
            continue
        heads.append(glyph)
    if not heads:
        return []
    first = heads[0]["x"]
    signs = [
        g for g in glyphs if lift_of_glyph(g["code"]) and g["x"] >= first - REACH * step
    ]
    notes = []
    for head in heads:
        y = head["origin"][1]
        before = [
            s
            for s in signs
            if head["x"] - REACH * step <= s["x"] < head["x"]
            and abs(s["origin"][1] - y) < step / 2
        ]
        sign = max(before, key=lambda s: s["x"]) if before else None
        notes.append(
            {
                "x": head["x"],
                "y": y,
                "pitch": pitch_at(y, bottom, step),
                "size": head["size"],
                "code": head["code"],
                "lift": lift_of_glyph(sign["code"]) if sign else None,
                "sign": sign,
            }
        )
    return notes


def is_grace(rhythm: str) -> bool:
    return rhythm.startswith("note") and rhythm.endswith("G")


def full_size(heads: list[dict]) -> list[dict]:
    """The heads printed at full size: a grace note is printed small, and not
    always with a glyph read here as a notehead."""
    largest = max((head["size"] for head in heads), default=0)
    return [head for head in heads if head["size"] >= 0.85 * largest]


def pair_notes(notes: list[tuple[str, str]], heads: list[dict]) -> list[int] | None:
    """For each (rhythm, pitch) note, the index of its head -- None for a grace
    note left out -- or None altogether if the two cannot be lined up.

    Lined up means the same count and, one by one, the same pitch.
    """
    def same(chosen_notes: list[int], chosen_heads: list[dict]) -> bool:
        return len(chosen_notes) == len(chosen_heads) and all(
            notes[n][1] == head["pitch"] for n, head in zip(chosen_notes, chosen_heads)
        )

    if not notes:
        return None
    everything = list(range(len(notes)))
    if same(everything, heads):
        return everything
    kept = [n for n in everything if not is_grace(notes[n][0])]
    big = full_size(heads)
    if not same(kept, big):
        return None
    where = {n: heads.index(head) for n, head in zip(kept, big)}
    return [where.get(n) for n in everything]


def count_heads(page: "fitz.Page", staff: list[float]) -> int:
    """How many full-size notes the page prints on this staff."""
    return len(full_size(read_staff(page, staff)))
