"""Does homr find the staffs on a Turkish score page?

Everything measured so far was given staffs cut out by this project, from the
coordinates inside the pdf -- which is to say, cut perfectly. A photograph comes
with no coordinates. homr has its own stage for this: a segmentation network
labels every pixel, and the staff lines, noteheads and barlines it finds are
assembled into staffs. That stage was trained on Western scores and has never
been tried on a Mus2 page.

This renders pages from the test split, hands each to homr's staff detection,
and compares what it finds against the staffs the pdf says are there. No reading
model is involved, so whatever it reports belongs to this stage alone.

With --photo each page is first made to look photographed: set on a table,
tilted, seen at an angle, lit unevenly, slightly blurred, noisy and saved as a
jpeg. The corners of every real staff go through the same warp, so the answer
key moves with the picture.

A detection belongs to the real staff whose middle is nearest its own, if that
is within one staff height. Three things can go wrong and each is counted: a
real staff nobody found, a real staff found twice, and a detection with no real
staff under it (lyrics or a title taken for a staff). A staff found but cut
short is caught too, by the share of its width the detection covers.

    python -m training.check_staff_detection --limit 5 --photo
"""

import argparse
import os
import random
import re
import tempfile
from concurrent.futures import Future

import albumentations as A
import cv2
import fitz
import numpy as np

from homr import download_utils
from homr import main as homr_main
from homr.main import ProcessingConfig, detect_staffs_in_image
from homr.segmentation.config import segnet_path_onnx
from homr.simple_logging import eprint
from training.datasets.convert_symbtr import symbtr_pdf, symbtr_test_index

STAFF = re.compile(r"-\d+$")
WEIGHTS = "https://github.com/liebharc/homr/releases/download/onnx_checkpoints/"

# homr scales every page to this width before segmenting it. Handing it pages
# already that wide means its coordinates need no conversion.
PAGE_WIDTH = 1920

# A phone photo of an A4 page is roughly this wide before homr shrinks it.
PHOTO_WIDTH = 2600


def download_segmentation() -> None:
    """Only the segmentation network; the reading model is not needed here."""
    if os.path.exists(segnet_path_onnx):
        return
    base_name = os.path.basename(segnet_path_onnx).split(".")[0]
    archive = os.path.join(os.path.dirname(segnet_path_onnx), base_name + ".zip")
    try:
        download_utils.download_file(WEIGHTS + base_name + ".zip", archive)
        download_utils.unzip_file(archive, os.path.dirname(segnet_path_onnx))
    finally:
        if os.path.exists(archive):
            os.remove(archive)


def _no_title(*_: object) -> Future:
    """Staff detection also starts reading the title; that is not measured here."""
    done: Future = Future()
    done.set_result("")
    return done


def real_staffs(page: "fitz.Page") -> list[tuple[float, float, float, float]]:  # noqa: F821
    """(top, bottom, left, right) of every staff the pdf draws, in pdf points.

    Mus2 draws the five lines of a staff as one path; the lines are its long
    horizontal items. Grouped the way convert_symbtr groups them.
    """
    width = page.rect.width
    rules = []
    for drawing in page.get_drawings():
        for item in drawing["items"]:
            if item[0] == "l":
                start, end = item[1], item[2]
                if abs(start.y - end.y) < 0.6 and abs(end.x - start.x) > width * 0.5:
                    left, right = sorted((start.x, end.x))
                    rules.append(((start.y + end.y) / 2, left, right))
            elif item[0] == "re":
                rect = item[1]
                if rect.height < 1.2 and rect.width > width * 0.5:
                    rules.append((rect.y0, rect.x0, rect.x1))
    rules.sort()
    staffs, current = [], rules[:1]
    for rule in rules[1:]:
        if rule[0] - current[-1][0] < 14:
            current.append(rule)
        else:
            staffs.append(current)
            current = [rule]
    if current:
        staffs.append(current)
    return [
        (group[0][0], group[-1][0], min(r[1] for r in group), max(r[2] for r in group))
        for group in staffs
        if len({round(r[0], 1) for r in group}) >= 4
    ]


def corners(staff: tuple[float, float, float, float], zoom: float) -> list[tuple[float, float]]:
    top, bottom, left, right = (value * zoom for value in staff)
    return [(left, top), (right, top), (right, bottom), (left, bottom)]


def _uneven_light(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """Darken one side of the page, as a lamp or a hand's shadow would."""
    height, width = image.shape[:2]
    angle = rng.uniform(0, 2 * np.pi)
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    ramp = (xs / width - 0.5) * np.cos(angle) + (ys / height - 0.5) * np.sin(angle)
    ramp = (ramp - ramp.min()) / (ramp.max() - ramp.min())
    darkest = rng.uniform(0.55, 0.8)
    light = darkest + (1 - darkest) * ramp
    return np.clip(image.astype(np.float32) * light[..., None], 0, 255).astype(np.uint8)


def photograph(
    image: np.ndarray, staffs: list[list[tuple[float, float]]], seed: int
) -> tuple[np.ndarray, list[list[tuple[float, float]]]]:
    """Make a clean render look like a phone photo; move the staff corners along."""
    rng = random.Random(seed)
    height, width = image.shape[:2]
    table = tuple(int(v) for v in rng.choice([(60, 70, 80), (120, 100, 80), (40, 40, 40)]))
    margins = [int(rng.uniform(0.03, 0.08) * width) for _ in range(4)]
    image = cv2.copyMakeBorder(image, *margins, cv2.BORDER_CONSTANT, value=table)
    shifted = [[(x + margins[2], y + margins[0]) for x, y in staff] for staff in staffs]

    image = _uneven_light(image, rng)
    warp = A.Compose(
        [
            A.Perspective(
                scale=(0.02, 0.05), fit_output=True, border_mode=cv2.BORDER_CONSTANT,
                fill=table, p=1.0,
            ),
            # fit_output, so that tilting never pushes the page's corners out of
            # the frame and cuts off the ends of a staff.
            A.Affine(
                rotate=(-3, 3), fit_output=True, border_mode=cv2.BORDER_CONSTANT,
                fill=table, p=1.0,
            ),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.5, 1.2), p=1.0),
                    A.MotionBlur(blur_limit=(3, 5), p=1.0),
                ],
                p=0.8,
            ),
            A.GaussNoise(std_range=(0.01, 0.04), p=1.0),
            A.ImageCompression(quality_range=(55, 85), p=1.0),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        seed=seed,
    )
    flat = [point for staff in shifted for point in staff]
    result = warp(image=image, keypoints=flat)
    image, moved = result["image"], [tuple(p[:2]) for p in result["keypoints"]]
    return image, [moved[i : i + 4] for i in range(0, len(moved), 4)]


def to_width(
    image: np.ndarray, staffs: list[list[tuple[float, float]]], width: int
) -> tuple[np.ndarray, list[list[tuple[float, float]]]]:
    scale = width / image.shape[1]
    resized = cv2.resize(image, (width, round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    return resized, [[(x * scale, y * scale) for x, y in staff] for staff in staffs]


def page_image(
    page: "fitz.Page", real: list, seed: int, photo: bool  # noqa: F821
) -> tuple[np.ndarray, list[list[tuple[float, float]]]]:
    """The page as homr will see it, and the corners of each real staff on it."""
    render_width = PHOTO_WIDTH if photo else PAGE_WIDTH
    zoom = render_width / page.rect.width
    pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    image = np.frombuffer(pixmap.samples, np.uint8).reshape(
        pixmap.height, pixmap.width, pixmap.n
    )[..., :3][..., ::-1].copy()
    staffs = [corners(staff, zoom) for staff in real]
    if photo:
        image, staffs = photograph(image, staffs, seed)
    return to_width(image, staffs, PAGE_WIDTH)


def page_seed(work: str, number: int) -> int:
    return sum(ord(c) for c in work) * 100 + number


def pair(real: list[list[tuple[float, float]]], found: list) -> tuple[dict[int, list[int]], int]:
    """Give each detection to the real staff whose middle is nearest its own.

    Returns the detections each real staff received, and how many detections
    had no real staff within one staff height.
    """
    middles = [np.mean([y for _, y in staff]) for staff in real]
    heights = [
        np.hypot(staff[3][0] - staff[0][0], staff[3][1] - staff[0][1]) for staff in real
    ]
    owner: dict[int, list[int]] = {index: [] for index in range(len(real))}
    stray = 0
    for number, detected in enumerate(found):
        middle = (detected[0] + detected[1]) / 2
        nearest = min(range(len(real)), key=lambda i: abs(middles[i] - middle))
        if abs(middles[nearest] - middle) <= heights[nearest]:
            owner[nearest].append(number)
        else:
            stray += 1
    return owner, stray


def compare(real: list[list[tuple[float, float]]], found: list) -> dict:
    """How the detections line up with the real staffs."""
    owner, stray = pair(real, found)
    coverage = []
    for i, hits in owner.items():
        if hits:
            left = min(x for x, _ in real[i])
            right = max(x for x, _ in real[i])
            detected = found[hits[0]]
            overlap = min(right, detected[3]) - max(left, detected[2])
            coverage.append(max(0.0, overlap) / (right - left))
    found_real = sum(1 for hits in owner.values() if hits)
    return {
        "found": found_real,
        "missed": len(real) - found_real,
        "doubled": sum(1 for hits in owner.values() if len(hits) > 1),
        "stray": stray,
        "coverage": coverage,
    }


def test_works() -> list[str]:
    works = []
    with open(symbtr_test_index, encoding="utf-8") as handle:
        for line in handle:
            name = os.path.basename(line.strip().split(",")[0])[: -len(".png")]
            work = STAFF.sub("", name)
            if work not in works:
                works.append(work)
    return works


def main() -> None:  # noqa: PLR0915
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Check only N works.")
    parser.add_argument("--pdf", default=symbtr_pdf, help="Folder holding the pdfs.")
    parser.add_argument("--photo", action="store_true", help="Make each page look photographed.")
    parser.add_argument("--keep", default=None, help="Save the pages with trouble here.")
    options = parser.parse_args()

    download_segmentation()
    homr_main.detect_title = _no_title
    config = ProcessingConfig(
        enable_debug=False,
        enable_cache=False,
        write_staff_positions=False,
        read_staff_positions=False,
        selected_staff=-1,
        use_gpu_inference=False,
    )

    works = test_works()
    if options.limit:
        works = works[: options.limit]
    kind = "photographed" if options.photo else "clean"
    eprint(f"Checking the pages of {len(works)} test works, {kind}")
    if options.keep:
        os.makedirs(options.keep, exist_ok=True)

    totals = {"real": 0, "found": 0, "missed": 0, "doubled": 0, "stray": 0}
    coverage: list[float] = []
    pages = clean_pages = failed = 0
    notes = []

    with tempfile.TemporaryDirectory() as scratch:
        for work in works:
            path = os.path.join(options.pdf, work + ".pdf")
            if not os.path.exists(path):
                continue
            with fitz.open(path) as document:
                for number, page in enumerate(document):
                    real = real_staffs(page)
                    if not real:
                        continue
                    image, staffs = page_image(
                        page, real, page_seed(work, number), options.photo
                    )
                    image_path = os.path.join(scratch, "page.jpg" if options.photo else "page.png")
                    cv2.imwrite(image_path, image)

                    pages += 1
                    totals["real"] += len(real)
                    label = f"{work} p{number + 1}"
                    try:
                        multi_staffs, _, _, _ = detect_staffs_in_image(image_path, config)
                    except Exception as error:  # noqa: BLE001
                        failed += 1
                        totals["missed"] += len(real)
                        notes.append(f"{label}: detection failed ({error})")
                        continue
                    found = [
                        (staff.min_y, staff.max_y, staff.min_x, staff.max_x)
                        for multi in multi_staffs
                        for staff in multi.staffs
                    ]
                    result = compare(staffs, found)
                    for key in ("found", "missed", "doubled", "stray"):
                        totals[key] += result[key]
                    coverage += result["coverage"]
                    if not (result["missed"] or result["stray"] or result["doubled"]):
                        clean_pages += 1
                        continue
                    notes.append(
                        f"{label}: {len(real)} staffs, found {result['found']},"
                        f" missed {result['missed']}, stray {result['stray']},"
                        f" twice {result['doubled']}"
                    )
                    if options.keep:
                        for top, bottom, left, right in found:
                            cv2.rectangle(
                                image, (int(left), int(top)), (int(right), int(bottom)),
                                (0, 0, 255), 3,
                            )
                        # cv2.imwrite cannot open a path with non-ascii letters
                        # on Windows, and the Drive folder is called "Drive'ım".
                        encoded = cv2.imencode(".jpg", image)[1]
                        with open(os.path.join(options.keep, f"{label}.jpg"), "wb") as handle:
                            handle.write(encoded.tobytes())

    real_total = totals["real"] or 1
    eprint(f"\npages ({kind})              : {pages}, every staff right on {clean_pages}")
    eprint(f"pages where detection fails: {failed}")
    eprint(f"real staffs                : {totals['real']}")
    found_share = 100 * totals["found"] / real_total
    eprint(f"  found                    : {totals['found']} = {found_share:.1f}%")
    eprint(f"  missed                   : {totals['missed']}")
    eprint(f"  found twice              : {totals['doubled']}")
    eprint(f"detections on no staff     : {totals['stray']}")
    if coverage:
        short = sum(1 for share in coverage if share < 0.95)
        eprint(f"width covered, on average  : {100 * sum(coverage) / len(coverage):.1f}%")
        eprint(f"  staffs cut short (<95%)  : {short}")
    if notes:
        eprint("\npages with trouble:")
        for note in notes:
            eprint(f"   {note}")


if __name__ == "__main__":
    main()
