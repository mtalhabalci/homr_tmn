"""Written tempo marks (a note value, '=', a number: an eighth = 112) read off a page.

A tempo mark stands above the staff it starts on; the header's mark stands above the first staff. For every staff
the band above it is cleaned of long lines and its ink grouped into clusters, roughly one per word or mark; a small
CRNN reads each cluster as 'U8 1 1 2' (note value, digits) or as nothing. The reader is trained on the printed
marks of the SymbTr Mus2 scores (whose .mu2 files hold every tempo) and on synthetic marks in printed and
handwriting-like fonts (training/tempo/).

read_tempo_marks(gray_page, staff_boxes) -> [TempoMark]; the weights come from OMR_TEMPO_MODEL (default
training/architecture/tempo/tempo_okuyucu.pt); without them no tempo is read.
"""
import os
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torch import nn

CROP_HEIGHT = 48
MAX_CROP_WIDTH = 640
TOKENS = ["U1", "U2", "U4", "U8", "U16", "U2d", "U4d", "U8d"] + list("0123456789")
UNIT_NAMES = {"U1": "whole", "U2": "half", "U4": "quarter", "U8": "eighth", "U16": "16th"}
MIN_CONFIDENCE = 0.6
WEIGHTS = os.environ.get("OMR_TEMPO_MODEL", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training", "architecture", "tempo", "tempo_okuyucu.pt"))


class TempoReader(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        def block(cin: int, cout: int, pool: int | tuple[int, int]) -> list[nn.Module]:
            return [nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True), nn.MaxPool2d(pool)]

        self.cnn = nn.Sequential(*block(1, 32, 2), *block(32, 64, 2), *block(64, 128, (2, 1)), *block(128, 192, (2, 1)),
                                 nn.Conv2d(192, 256, (3, 1)), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.rnn = nn.LSTM(256, 128, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2)
        self.out = nn.Linear(256, len(TOKENS) + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, 1, 48, W) in [0, 1], ink dark
        f = self.cnn(1.0 - x).squeeze(2).permute(0, 2, 1)
        f, _ = self.rnn(f)
        return self.out(f)


@dataclass
class TempoMark:
    staff: int  # index into the staff boxes given
    box: tuple[int, int, int, int]  # on the page
    unit: str  # MusicXML beat unit: 'eighth', 'quarter', ...
    dotted: bool
    per_minute: int
    confidence: float


def bands(staffs: list[tuple[int, int, int, int]], page_width: int, up: float = 3.5) -> list[tuple[int, int, int, int]]:
    """The band above every staff: from up staff heights above its top line (not into the staff above; for the
    first staff from the top of the page, where the header's mark is) down to a quarter staff height below it."""
    out = []
    for k, (_, y0, _, y1) in enumerate(staffs):
        height = max(y1 - y0, 8)
        top = 0 if k == 0 else max(y0 - up * height, staffs[k - 1][3] + 0.25 * height)
        out.append((0, int(max(0, top)), page_width, int(y0 + 0.25 * height)))
    return out


def clusters(gray: np.ndarray, band: tuple[int, int, int, int], staff_height: int) -> list[tuple[int, int, int, int]]:
    """Ink clusters in a band, in page coordinates: long horizontal lines removed, ink joined sideways only (two
    text lines stacked closely, like the header's usul line over its tempo line, stay apart)."""
    x0, y0, x1, y1 = band
    if y1 - y0 < 4:
        return []
    part = (gray[y0:y1, x0:x1] < 150).astype(np.uint8) * 255
    space = max(staff_height / 4, 2)
    long_line = cv2.morphologyEx(part, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (int(4 * space), 1)))
    part = cv2.subtract(part, long_line)
    joined = cv2.dilate(part, cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(1.3 * space)), 1)))
    count, _, stats, _ = cv2.connectedComponentsWithStats(joined, 8)
    out = []
    for i in range(1, count):
        cx, cy, cw, ch, area = stats[i]
        if ch < 0.6 * space or cw < 0.6 * space or area < 2 * space * space:
            continue
        if ch > 3.5 * staff_height or cw > 0.7 * (x1 - x0):
            continue
        out.append((x0 + cx, y0 + cy, x0 + cx + cw, y0 + cy + ch))
    return out


def crop(gray: np.ndarray, box: tuple[int, int, int, int], staff_height: int) -> np.ndarray:
    """A cluster cut with a margin and scaled to CROP_HEIGHT (width in proportion, at most MAX_CROP_WIDTH)."""
    margin = int(max(staff_height / 8, 2))
    x0, y0, x1, y1 = box
    h, w = gray.shape
    piece = gray[max(0, y0 - margin):min(h, y1 + margin), max(0, x0 - margin):min(w, x1 + margin)]
    scale = CROP_HEIGHT / piece.shape[0]
    width = int(min(MAX_CROP_WIDTH, max(8, round(piece.shape[1] * scale))))
    return cv2.resize(piece, (width, CROP_HEIGHT), interpolation=cv2.INTER_AREA)


def to_tensor(images: list[np.ndarray]) -> torch.Tensor:
    width = max(16, (max(im.shape[1] for im in images) + 3) // 4 * 4)
    batch = np.full((len(images), 1, CROP_HEIGHT, width), 255, np.uint8)
    for i, im in enumerate(images):
        batch[i, 0, :, :im.shape[1]] = im
    return torch.from_numpy(batch).float() / 255.0


def decode(logits: torch.Tensor) -> list[tuple[list[str], float]]:
    """Greedy CTC: per item the tokens and a confidence (the mean probability of the tokens chosen)."""
    probs = logits.softmax(-1)
    best = probs.argmax(-1)
    out = []
    for b in range(best.shape[0]):
        seq, conf, prev = [], [], 0
        for t, k in enumerate(best[b].tolist()):
            if k not in (0, prev):
                seq.append(TOKENS[k - 1])
                conf.append(probs[b, t, k].item())
            prev = k
        out.append((seq, float(np.mean(conf)) if conf else 0.0))
    return out


def valid(seq: list[str]) -> bool:
    """A reading is a tempo mark only as one note value followed by 2-4 digits."""
    return len(seq) >= 3 and seq[0].startswith("U") and all(t.isdigit() for t in seq[1:]) and len(seq) <= 5


_model: TempoReader | None = None


def _load() -> TempoReader | None:
    global _model  # noqa: PLW0603
    if _model is None and os.path.exists(WEIGHTS):
        model = TempoReader()
        model.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
        _model = model.eval()
    return _model


@torch.no_grad()
def read_tempo_marks(gray: np.ndarray, staffs: list[tuple[int, int, int, int]]) -> list[TempoMark]:
    """The tempo marks above the given staffs (boxes of their lines, top to bottom, on this gray page)."""
    model = _load()
    if model is None or not staffs:
        return []
    found = []
    for k, band in enumerate(bands(staffs, gray.shape[1])):
        height = staffs[k][3] - staffs[k][1]
        boxes = clusters(gray, band, height)
        if not boxes:
            continue
        readings = decode(model(to_tensor([crop(gray, b, height) for b in boxes])))
        for box, (seq, confidence) in zip(boxes, readings, strict=True):
            if not valid(seq) or confidence < MIN_CONFIDENCE:
                continue
            value = int("".join(seq[1:]))
            if not 20 <= value <= 1200:
                continue
            unit = seq[0].rstrip("d")
            found.append(TempoMark(k, box, UNIT_NAMES[unit], seq[0].endswith("d"), value, round(confidence, 3)))
    return found
