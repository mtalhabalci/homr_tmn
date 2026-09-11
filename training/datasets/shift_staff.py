"""Re-engrave a staff a few degrees higher or lower, in Mus2's own hand.

The corpus places some signs on one degree and nowhere else: 752 of the 758
five-comma sharps sit on F5 and six on F4, and the model reads the first at 89%
and the second at 54%. It learns the sign together with its height, because
nothing in the data asks it not to.

Drawing new pages from scratch would mean making every engraving decision Mus2
makes. This does not: every glyph's code, position and size can be read back out
of the pdf and the font is embedded in it, so a staff can be rebuilt exactly as
it stands, and rebuilt again with everything moved down two degrees. Mus2's
spacing, stems and beams come along untouched.

What moves and what does not:

    clef, time signature   stay -- the clef is what fixes pitch to height
    key signature, notes   move -- their relation to each other is preserved
    staff lines            stay -- they are the grid
    ledger lines           avoided: staffs that need any are left out

The result is not music any more: a rast piece moved two degrees is not in rast.
That does not matter here. What the model is being taught is what a sign looks
like, and the picture still matches its labels exactly.
"""

import os

from homr.simple_logging import eprint
from homr.transformer.vocabulary import EncodedSymbol
from training.datasets.convert_symbtr import (
    ACCIDENTAL_GLYPHS,
    NOTEHEAD_GLYPHS,
    _staff_lines,
    symbtr_pdf,
    target_page_width,
)

LETTERS = "CDEFGAB"
MARGIN = 34
LEDGER_HALF_WIDTH = 3.6


def shift_pitch(pitch: str, steps: int) -> str | None:
    """Move a pitch name up or down whole degrees. C4 down one is B3."""
    if len(pitch) < 2 or pitch[0] not in LETTERS or not pitch[1:].isdigit():
        return None
    index = LETTERS.index(pitch[0]) + 7 * int(pitch[1:])
    moved = index + steps
    octave, letter = divmod(moved, 7)
    if not 0 <= octave <= 9:
        return None
    return f"{LETTERS[letter]}{octave}"


def shift_tokens(rows: list[EncodedSymbol], steps: int) -> list[EncodedSymbol] | None:
    """The same symbols, every pitch moved. Anything else is untouched."""
    moved = []
    for symbol in rows:
        if symbol.rhythm == "clef_G2" or symbol.pitch in (".", "_"):
            moved.append(symbol)
            continue
        pitch = shift_pitch(symbol.pitch, steps)
        if pitch is None:
            return None
        copied = symbol.change_lift(symbol.lift)
        copied.pitch = pitch
        moved.append(copied)
    return moved


def _characters(page, band):  # noqa: ANN001, ANN202
    """Every character drawn in the band, notation and lyrics alike.

    The lyrics matter as much as the notes here: a staff rebuilt without them
    would differ from a real one in a way the model could learn to spot, and
    then the test would be measuring the rebuild rather than the shift.
    """
    found = []
    for block in page.get_text("rawdict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                font = span.get("font", "")
                for char in span.get("chars", []):
                    x0, y0, x1, y1 = char["bbox"]
                    if band.y0 <= (y0 + y1) / 2 <= band.y1:
                        found.append(
                            {
                                "origin": char["origin"],
                                "char": char["c"],
                                "code": ord(char["c"]),
                                "size": span["size"],
                                "font": font,
                                "notation": "Mus2" in font,
                                "x": x0,
                                "mid": (y0 + y1) / 2,
                            }
                        )
    return sorted(found, key=lambda g: g["x"])


def _is_staff_line(y: float, staff: list[float]) -> bool:
    return any(abs(y - line) < 0.8 for line in staff)


def _face_for(faces, glyph):  # noqa: ANN001, ANN202
    """The embedded subset that carries this character."""
    wanted = glyph["font"]
    for name, handle, font in faces:
        if name == wanted and font.has_glyph(glyph["code"]):
            return handle
    for name, handle, font in faces:
        if font.has_glyph(glyph["code"]):
            return handle
    return None


def _in_band(item, band) -> bool:  # noqa: ANN001
    points = [p for p in item[1:] if hasattr(p, "y")]
    if not points:
        box = item[1]
        return band.y0 <= box.y0 <= band.y1 if hasattr(box, "y0") else False
    return all(band.y0 - 2 <= point.y <= band.y1 + 2 for point in points)


def _moved(point, dy):  # noqa: ANN001, ANN202
    import fitz  # noqa: PLC0415

    return fitz.Point(point.x, point.y + dy)


def _holds_still(item, staff) -> bool:  # noqa: ANN001
    """True for a staff line, which is the grid the music is read against.

    The decision has to be made for each line and not for the path it belongs
    to: Mus2 draws all five lines of a staff as one path, so a path-level test
    sees something 21 points tall, calls it music, and carries the whole staff
    along with the notes. The notes then sit exactly where they sat before, the
    pitch is unchanged, and only the labels have moved.
    """
    if item[0] != "l":
        return False
    start, end = item[1], item[2]
    if abs(start.y - end.y) >= 0.4:
        return False
    return _is_staff_line((start.y + end.y) / 2, staff)


def render_shifted(stem: str, staff_index: int, steps: int, destination: str) -> bool:
    """Draw one staff again with the music moved. False if it cannot be done cleanly."""
    import fitz  # noqa: PLC0415

    path = os.path.join(symbtr_pdf, stem + ".pdf")
    if not os.path.exists(path):
        return False

    with fitz.open(path) as document:
        staves = [(page, lines) for page in document for lines in _staff_lines(page)]
        if staff_index >= len(staves):
            return False
        page, staff = staves[staff_index]
        top, bottom = staff[0], staff[-1]
        gap = (bottom - top) / 4
        step = gap / 2
        band = fitz.Rect(0, top - MARGIN, page.rect.width, bottom + MARGIN)
        # A pdf counts y downwards and a staff counts degrees upwards, so a
        # note moved two degrees down is drawn two steps further down the page:
        # the two signs are opposite. Getting this backwards moves the picture
        # one way and its labels the other, and every pitch in the test is then
        # wrong by four degrees -- which reads exactly like a model that cannot
        # follow a sign off its usual line.
        offset = -steps * step

        glyphs = _characters(page, band)
        if not glyphs:
            return False
        first_note = next((g["x"] for g in glyphs if g["code"] in NOTEHEAD_GLYPHS), None)
        if first_note is None:
            return False

        def stays(glyph: dict) -> bool:
            # The clef and the time signature carry no pitch, so they hold their
            # place; the key signature's accidentals do carry one and move. So do
            # lyrics: they sit under the note they belong to, not at a height.
            if not glyph["notation"]:
                return True
            return glyph["x"] < first_note and glyph["code"] not in ACCIDENTAL_GLYPHS

        # A path counts if any of it falls in the band; its own items are
        # filtered below. Testing the top edge alone dropped paths that begin
        # higher up the page and reach down into this staff.
        drawings = [
            d
            for d in page.get_drawings()
            if d["rect"].y1 >= band.y0 and d["rect"].y0 <= band.y1
        ]

        # get_fonts reports the subset name a pdf gives a font, "PBPBEM+Mus2";
        # the text itself reports the plain one, "Mus2". A page also carries
        # several subsets of the same face, each holding different letters, so
        # the right one is the one that has the letter -- a substitute font drops
        # the Turkish letters and "Aski" appears where "Aşkı" should.
        embedded = []
        for xref, _, _, base, _, _, _ in page.get_fonts(full=True):
            buffer = document.extract_font(xref)[3]
            if buffer:
                embedded.append((base.split("+")[-1], buffer))
        if not any(name == "Mus2" for name, _ in embedded):
            return False

        out = fitz.open()
        canvas = out.new_page(width=page.rect.width, height=band.height)
        faces = []
        for index, (name, buffer) in enumerate(embedded):
            handle = f"f{index}"
            canvas.insert_font(fontname=handle, fontbuffer=buffer)
            faces.append((name, handle, fitz.Font(fontbuffer=buffer)))

        for drawing in drawings:
            still, moving = [], []
            for item in drawing["items"]:
                if not _in_band(item, band):
                    continue
                (still if _holds_still(item, staff) else moving).append(item)
            for items, dy in ((still, -band.y0), (moving, -band.y0 + offset)):
                if not items:
                    continue
                shape = canvas.new_shape()
                for item in items:
                    if item[0] == "l":
                        shape.draw_line(_moved(item[1], dy), _moved(item[2], dy))
                    elif item[0] == "c":
                        shape.draw_bezier(*(_moved(point, dy) for point in item[1:5]))
                    elif item[0] == "re":
                        box = item[1]
                        shape.draw_rect(fitz.Rect(box.x0, box.y0 + dy, box.x1, box.y1 + dy))
                    elif item[0] == "qu":
                        shape.draw_quad(item[1] + (0, dy))
                shape.finish(
                    color=drawing.get("color"),
                    fill=drawing.get("fill"),
                    width=drawing.get("width") or 0.6,
                    closePath=drawing.get("closePath", False),
                )
                shape.commit()

        for glyph in glyphs:
            move = 0 if stays(glyph) else offset
            handle = _face_for(faces, glyph)
            if handle is None:
                continue
            canvas.insert_text(
                fitz.Point(glyph["origin"][0], glyph["origin"][1] - band.y0 + move),
                glyph["char"],
                fontname=handle,
                fontsize=glyph["size"],
                color=(0, 0, 0),
            )

        dpi = round(target_page_width / page.rect.width * 72)
        out[0].get_pixmap(dpi=dpi).save(destination)
        out.close()
    return True


def _degree(pitch: str) -> int | None:
    if len(pitch) < 2 or pitch[0] not in LETTERS or not pitch[1:].isdigit():
        return None
    return LETTERS.index(pitch[0]) + 7 * int(pitch[1:])


LOWEST_ON_STAFF = _degree("E4")
HIGHEST_ON_STAFF = _degree("F5")


def leaves_the_staff(rows: list[EncodedSymbol], steps: int) -> bool:
    """True if any note sits off the staff before or after the shift.

    A note off the staff rides on ledger lines, and the shift changes which
    notes need them: some would keep a line they no longer want, others would
    float with none. Rather than redraw them from a notehead's box -- which for
    most Mus2 glyphs holds the stem as well, so its middle is nowhere near the
    note -- staffs that need any are left out. The rest are exact.
    """
    for symbol in rows:
        if not symbol.rhythm.startswith("note"):
            continue
        before = _degree(symbol.pitch)
        moved = shift_pitch(symbol.pitch, steps)
        after = _degree(moved) if moved else None
        if before is None or after is None:
            return True
        if not LOWEST_ON_STAFF <= before <= HIGHEST_ON_STAFF:
            return True
        if not LOWEST_ON_STAFF <= after <= HIGHEST_ON_STAFF:
            return True
    return False


if __name__ == "__main__":
    eprint(__doc__)
