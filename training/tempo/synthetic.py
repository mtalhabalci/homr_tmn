"""Synthetic tempo-mark crops: a metronome note (SMuFL music fonts) with '=' and a number (printed and
handwriting-like text fonts), and look-alike non-marks (lone numbers, ending numbers, section words), drawn the
way homr.tempo_reader.crop cuts real clusters (CROP_HEIGHT high, width in proportion).

    python -m training.tempo.synthetic      writes datasets/tempo-mus2/synthetic_samples.png to look at

The text fonts are Windows fonts (C:\\Windows\\Fonts); missing ones are skipped.
"""
import os
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from homr.tempo_reader import CROP_HEIGHT, MAX_CROP_WIDTH

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Gootville and MScore have no metronome notes
MUSIC_FONTS = [os.path.join(ROOT, "datasets", "fonts", f) for f in
               ("Bravura.otf", "Leland.otf", "Petaluma.otf", "FinaleMaestro.otf", "FinaleBroadway.otf", "MuseJazz.otf")]
WIN = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
ARROW_FONT = os.path.join(WIN, "seguisym.ttf")
PRINT_FONTS = ["times.ttf", "timesbd.ttf", "arial.ttf", "arialbd.ttf", "georgia.ttf", "georgiab.ttf", "calibri.ttf",
               "calibrib.ttf", "cambria.ttc", "cour.ttf", "verdana.ttf", "tahoma.ttf"]
HAND_FONTS = ["Inkfree.ttf", "LHANDW.TTF", "mvboli.ttf", "comic.ttf", "comicbd.ttf", "Gabriola.ttf", "FRSCRIPT.TTF",
              "segoepr.ttf", "segoesc.ttf"]
TEXT_FONTS = [os.path.join(WIN, f) for f in PRINT_FONTS + HAND_FONTS if os.path.exists(os.path.join(WIN, f))]
# SMuFL metronome notes (stem up) and the augmentation dot
NOTE = {"U1": "\uECA2", "U2": "\uECA3", "U4": "\uECA5", "U8": "\uECA7", "U16": "\uECA9"}
DOT = "\uECB7"
ARROW = "\u21D2"
WORDS = ["SAZ", "Meyan", "Teslim", "Karar", "1. Hâne", "2. Hâne", "3. Hâne", "4. Hâne", "Usul: Aksak", "Usul: Düyek",
         "Mülâzime", "Zeyl", "Fine", "D.C.", "Başa", "(1)", "(2)", "1.", "2.", "Nakarat", "Ağır", "Yürük", "rit.",
         "accel.", "Usul: Sofyan", "Aranağme", "Serbest", "Ad lib.", "ten.", "3", "6", "5"]
SCALE = 4  # draw large, then shrink to the crop height


def _render(text: str, path: str, size: int) -> np.ndarray | None:
    """One part drawn alone and cut to its ink; None when the font drew nothing (a missing glyph)."""
    font = ImageFont.truetype(path, size)
    canvas = Image.new("L", (size * (len(text) + 4), size * 4), 255)
    ImageDraw.Draw(canvas).text((size, size), text, font=font, fill=0)
    a = np.array(canvas)
    ys, xs = np.where(a < 128)
    if len(xs) == 0:
        return None
    return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _draw_parts(parts: list[tuple[str, str, int]], rng: random.Random) -> np.ndarray | None:
    """Each part drawn alone, then set side by side on their bottoms (the note head sits on the digits' baseline).
    None when any part drew nothing."""
    pieces = []
    for text, path, size in parts:
        piece = _render(text, path, size)
        if piece is None:
            return None
        pieces.append(piece)
    height = max(p.shape[0] for p in pieces) + SCALE * 40
    width = sum(p.shape[1] for p in pieces) + SCALE * 10 * len(pieces) + 40
    canvas = np.full((height, width), 255, np.uint8)
    x, bottom = 20, height - SCALE * 20
    for piece in pieces:
        h, w = piece.shape
        y = bottom - h + rng.randint(-SCALE * 2, SCALE * 2)
        canvas[y:y + h, x:x + w] = np.minimum(canvas[y:y + h, x:x + w], piece)
        x += w + rng.randint(SCALE * 1, SCALE * 8)
    return canvas


def _finish(img: np.ndarray, rng: random.Random, staff_below: bool) -> np.ndarray:
    """Tight crop to the ink with a margin, light degradation, scale to CROP_HEIGHT."""
    if rng.random() < 0.5:
        m = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), rng.uniform(-3, 3), 1.0)
        img = cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), borderValue=255)
    ys, xs = np.where(img < 128)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    if staff_below:
        # a mid-piece mark sits on the staff: the top lines of a staff run under it
        gap = rng.randint(2, 12) * SCALE // 2
        space = max(8, (y1 - y0) // 3)
        for k in range(rng.randint(1, 2)):
            yy = y1 + gap + k * space
            if yy < img.shape[0] - 2:
                cv2.line(img, (0, yy), (img.shape[1] - 1, yy), 0, max(1, SCALE // 2))
        y1 = min(img.shape[0] - 1, y1 + gap + space)
    margin = int((y1 - y0) / 8) + SCALE
    img = img[max(0, y0 - margin):y1 + margin, max(0, x0 - margin):x1 + margin]
    if rng.random() < 0.4:
        k = np.ones((rng.choice([2, 3]),) * 2, np.uint8)
        img = cv2.erode(img, k) if rng.random() < 0.5 else cv2.dilate(img, k)
    scale = CROP_HEIGHT / img.shape[0]
    width = int(min(MAX_CROP_WIDTH, max(8, round(img.shape[1] * scale))))
    img = cv2.resize(img, (width, CROP_HEIGHT), interpolation=cv2.INTER_AREA)
    if rng.random() < 0.4:
        img = cv2.GaussianBlur(img, (3, 3), rng.uniform(0.3, 1.0))
    if rng.random() < 0.4:
        noise = np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, rng.uniform(4, 18), img.shape)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def sample(rng: random.Random) -> tuple[np.ndarray, str]:
    """One synthetic crop and its target ('U8 1 6 8', or '' for a non-mark)."""
    while True:
        text_font = rng.choice(TEXT_FONTS)
        size = SCALE * rng.randint(9, 15)
        staff_below = rng.random() < 0.2
        if rng.random() < 0.7:
            unit = rng.choices(list(NOTE), weights=[1, 4, 40, 40, 6])[0]
            dotted = rng.random() < 0.08 and unit in ("U2", "U4", "U8")
            value = rng.choice([rng.randint(40, 200), rng.randint(40, 400), rng.randint(20, 999)])
            parts = [(NOTE[unit] + (DOT if dotted else ""), rng.choice(MUSIC_FONTS), int(size * rng.uniform(1.6, 2.6)))]
            parts.append((rng.choice(["=", "= ", " = "]) + str(value), text_font, size))
            if rng.random() < 0.25:
                parts.append((ARROW, ARROW_FONT, size))
                parts.append((rng.choice(["1 Dk 7 Sn", "47 Sn", "3 Dk 29 Sn", "2 Dk"]), text_font, size))
            if rng.random() < 0.1:
                parts.insert(0, (rng.choice(["M.M.", "Tempo", "("]), text_font, size))
            target = (unit + ("d" if dotted else "")) + " " + " ".join(str(value))
        else:
            # look-alikes: words, lone numbers, a note without a number
            choice = rng.random()
            if choice < 0.6:
                parts = [(rng.choice(WORDS), text_font, size)]
            elif choice < 0.85:
                parts = [(str(rng.randint(1, 400)), text_font, size)]
            else:
                parts = [(rng.choice(list(NOTE.values())), rng.choice(MUSIC_FONTS), int(size * 2))]
            target = ""
        drawn = _draw_parts(parts, rng)
        if drawn is not None:
            return _finish(drawn, rng, staff_below), target


if __name__ == "__main__":
    rng = random.Random(1)
    rows = []
    for _ in range(24):
        img, target = sample(rng)
        canvas = np.full((70, 660), 255, np.uint8)
        canvas[:CROP_HEIGHT, :min(660, img.shape[1])] = img[:, :660]
        cv2.putText(canvas, target or "-", (5, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
        rows.append(canvas)
    os.makedirs(os.path.join(ROOT, "datasets", "tempo-mus2"), exist_ok=True)
    cv2.imwrite(os.path.join(ROOT, "datasets", "tempo-mus2", "synthetic_samples.png"), np.vstack(rows))
