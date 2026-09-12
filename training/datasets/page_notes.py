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
# Mus2 has two dot characters; which one depends on where the dot must sit.
DOTS = frozenset({0xE2, 0xAB})
# What the notehead character says about the note's value: a whole, a half,
# a quarter, or shorter -- beamed or flagged, which the head alone cannot split.
HEAD_KIND = {0x28: "1", 0x29: "2", 0x27: "2", 0x2A: "4", 0x6F: "4",
             0x78: "short", 0x2B: "short", 0x25: "short", 0x2C: "short"}

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
    dots = [g for g in glyphs if g["code"] in DOTS]
    notes = []
    for head in heads:
        y = head["origin"][1]
        dotted = any(
            0 < dot["x"] - head["origin"][0] < 4 * step and abs(dot["origin"][1] - y) < 1.5 * step
            for dot in dots
        )
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
                "origin": head["origin"],
                "y": y,
                "pitch": pitch_at(y, bottom, step),
                "size": head["size"],
                "code": head["code"],
                "kind": HEAD_KIND.get(head["code"]),
                "dotted": dotted,
                "lift": lift_of_glyph(sign["code"]) if sign else None,
                "sign": sign,
            }
        )
    return notes


def key_signature(page: "fitz.Page", staff: list[float], heads: list[dict]) -> list[tuple[str, str]]:
    """The key signature at the head of the staff, as (pitch, sign), left to right.

    The signs standing left of the first notehead and out of reach of that
    note's own sign. Mus2 sometimes draws a signature sign twice on the same
    spot; it shows once, so it counts once.
    """
    if not heads:
        return []
    top, bottom = staff[0], staff[-1]
    step = (bottom - top) / 8
    band = fitz.Rect(0, top - MARGIN, page.rect.width, bottom + MARGIN)
    limit = heads[0]["x"] - REACH * step
    found: list[dict] = []
    for glyph in _characters(page, band):
        lift = lift_of_glyph(glyph["code"]) if glyph["notation"] else None
        if not lift or lift == "N" or glyph["x"] >= limit:
            continue
        y = glyph["origin"][1]
        if not top - 4 * step <= y <= bottom + 4 * step:
            continue
        if any(g["code"] == glyph["code"] and abs(g["x"] - glyph["x"]) < 1 and abs(g["origin"][1] - y) < 1
               for g in found):
            continue
        found.append(glyph)
    found.sort(key=lambda g: g["x"])
    return [(pitch_at(g["origin"][1], bottom, step), lift_of_glyph(g["code"])) for g in found]


# The mark Mus2 closes some lines with: a heavy rule with the program's name
# set sideways against it. It stands where a barline would, and reads as one.
LINE_END = 0x178


def bar_rules(page: "fitz.Page", staff: list[float], heads: list[dict]) -> list[tuple[float, str]]:
    """The barline strokes on a staff, as (x, "thin" | "thick" | "dot").

    A barline spans the staff exactly, and a stem touches a notehead; that is
    how the two are told apart. Mus2 draws the thick stroke of a repeat as a
    filled rectangle or a wide line, and each repeat dot as a small filled
    circle of four arcs in one of the two middle spaces. (The segno, 0x60, is
    a character above the staff and no barline at all.)
    """
    top, bottom = staff[0], staff[-1]
    height = bottom - top
    step = height / 8
    strokes = []
    arcs = []
    for drawing in page.get_drawings():
        width = drawing.get("width") or 0
        for item in drawing["items"]:
            if item[0] == "l":
                a, b = item[1], item[2]
                if abs(a.x - b.x) < 0.9 and abs(min(a.y, b.y) - top) < 2.5 and abs(max(a.y, b.y) - bottom) < 2.5:
                    strokes.append(((a.x + b.x) / 2, "thick" if width >= 1.5 else "thin"))
            elif item[0] == "re":
                r = item[1]
                if r.height > height * 0.8 and abs(r.y0 - top) < 2.5 and abs(r.y1 - bottom) < 2.5 and r.width < 4:
                    strokes.append(((r.x0 + r.x1) / 2, "thick" if r.width >= 1.5 else "thin"))
            elif item[0] == "c" and drawing.get("fill") is not None:
                xs = [p.x for p in item[1:5]]
                ys = [p.y for p in item[1:5]]
                if max(xs) - min(xs) < 1.2 and max(ys) - min(ys) < 1.2:
                    arcs.append(((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2))
    strokes = [
        (x, kind) for x, kind in strokes
        if not any(h["x"] - 2.5 <= x <= h["x"] + 2.5 * 3 for h in heads if kind == "thin")
    ]
    # Four arcs make a dot; a repeat dot sits in the second or third space.
    dots: list[tuple[float, float]] = []
    for x, y in arcs:
        if not staff[1] < y < staff[3]:
            continue
        if any(abs(x - dx) < 1.2 and abs(y - dy) < 1.2 for dx, dy in dots):
            continue
        dots.append((x, y))
    strokes += [(x, "dot") for x, _ in dots]
    band = fitz.Rect(0, top - MARGIN, page.rect.width, bottom + MARGIN)
    notation = [glyph for glyph in _characters(page, band) if glyph["notation"]]
    for glyph in notation:
        if glyph["code"] == LINE_END and top - step <= glyph["origin"][1] <= bottom + 4 * step:
            strokes.append((glyph["x"], "thin"))
    # Whatever stands left of the clef is the edge of the system, not a barline.
    clef = min((glyph["x"] for glyph in notation), default=0)
    strokes = [(x, kind) for x, kind in strokes if x > clef]
    strokes.sort()
    merged: list[tuple[float, str]] = []
    for x, kind in strokes:
        # A rule drawn twice on one spot is one rule. Two dots on one spot are
        # the upper and lower dot of a repeat, and both count.
        if kind != "dot" and merged and abs(merged[-1][0] - x) < 0.8 and merged[-1][1] == kind:
            continue
        merged.append((x, kind))
    return merged


def volta_hooks(page: "fitz.Page", staff: list[float]) -> list[tuple[float, str]]:
    """Where volta brackets open and close on a staff, as (x, "start" | "end").

    A bracket is a rule above the staff with a short hook turned down at the
    end where it begins, and another where it closes. A bracket carried over
    from the line before has no opening hook; one carried on to the next line
    has no closing hook.
    """
    top = staff[0]
    hooks, rules = [], []
    for drawing in page.get_drawings():
        # Bracket and hook are hairlines; a beam is drawn wide.
        if (drawing.get("width") or 0) >= 1:
            continue
        for item in drawing["items"]:
            if item[0] != "l":
                continue
            a, b = item[1], item[2]
            # The bracket rides higher over high notes and lower over low
            # ones, so its hooks run from 3 to 30 points. Mus2 draws tuplet
            # brackets as curves, so a straight hook up here is a volta's.
            if (
                abs(a.x - b.x) < 0.5
                and top - 45 < min(a.y, b.y) < top - 3
                and max(a.y, b.y) <= top + 1
                and 3 <= abs(a.y - b.y) <= 30
            ):
                hooks.append(((a.x + b.x) / 2, min(a.y, b.y)))
            # A short bracket over one measure leaves only a stub of rule
            # between its hook and its number.
            elif abs(a.y - b.y) < 0.5 and top - 45 < a.y < top - 3 and abs(a.x - b.x) > 0.8:
                rules.append((min(a.x, b.x), max(a.x, b.x), a.y))
    found = []
    for x, y in hooks:
        opens = any(abs(left - x) < 0.8 and abs(ry - y) < 0.8 for left, _, ry in rules)
        closes = any(abs(right - x) < 0.8 and abs(ry - y) < 0.8 for _, right, ry in rules)
        if opens:
            found.append((x, "start"))
        if closes:
            found.append((x, "end"))
    return sorted(found)


def bar_shape(strokes: list[str]) -> str:
    """What a run of strokes between two notes means, in label terms.

    plain, a repeat's "end", "start" or "end+start" -- told by which side its
    dots are on -- or "final", a heavy double bar with no dots at all.
    """
    rules = [kind for kind in strokes if kind != "dot"]
    if not rules:
        return "none"
    first = strokes.index(rules[0])
    last = len(strokes) - 1 - strokes[::-1].index(rules[-1])
    left = strokes[:first].count("dot") >= 2
    right = strokes[last + 1 :].count("dot") >= 2
    if left and right:
        return "end+start"
    if left:
        return "end"
    if right:
        return "start"
    return "final" if "thick" in rules else "plain"


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
