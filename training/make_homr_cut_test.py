"""Build a test set of staffs cut the way homr cuts them.

The model learned from staffs this project cut out of the pdf: the full page
width, 34 points above and below the lines, nothing straightened. On a
photograph there is no pdf; homr finds the staff itself, crops it tighter (four
line spacings above and below, never past the neighbouring staff), straightens
it, and hands that over. check_staff_detection showed homr finds nearly every
staff. Whether the model still reads what homr hands it is a separate question,
and this builds the material to answer it.

Each test page is rendered clean and, with the same seeds check_staff_detection
uses, made to look photographed. homr detects and cuts the staffs of each; a
cut is kept when it pairs with a real staff, and filed under that staff's name
so it is scored against the same labels as the original crop.

The pngs and the two index files go straight into one archive:

    python -m training.make_homr_cut_test --out "G:/.../homr_makam/homr_kesim_sinavi.tar.gz"

and are scored on Colab with evaluate_makam --clean, since the cuts from the
photographed pages carry their damage already.
"""

import argparse
import io
import os
import tarfile
import tempfile

import cv2
import fitz

from homr import main as homr_main
from homr.main import ProcessingConfig, detect_staffs_in_image
from homr.simple_logging import eprint
from homr.staff_parsing import prepare_staff_image
from homr.staff_regions import StaffRegions
from training.check_staff_detection import (
    STAFF,
    _no_title,
    download_segmentation,
    page_image,
    page_seed,
    pair,
    real_staffs,
    test_works,
)
from training.datasets.convert_symbtr import symbtr_pdf, symbtr_test_index

FOLDER = "datasets/SymbTr-homrcut"
KINDS = ("clean", "photo")


def _add(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    archive.addfile(info, io.BytesIO(data))


def main() -> None:  # noqa: PLR0915
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="The archive to write.")
    parser.add_argument("--limit", type=int, default=None, help="Only N works.")
    parser.add_argument("--pdf", default=symbtr_pdf, help="Folder holding the pdfs.")
    parser.add_argument("--samples", default=None, help="Also save a few cuts here, to look at.")
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

    with open(symbtr_test_index, encoding="utf-8") as handle:
        labelled = {
            os.path.basename(line.strip().split(",")[1])[: -len(".tokens")]
            for line in handle
            if line.strip()
        }

    works = test_works()
    if options.limit:
        works = works[: options.limit]
    if options.samples:
        os.makedirs(options.samples, exist_ok=True)

    index = {kind: [] for kind in KINDS}
    missed = {kind: 0 for kind in KINDS}
    failed = {kind: 0 for kind in KINDS}
    eprint(f"Cutting the staffs of {len(works)} test works, clean and photographed")

    with tempfile.TemporaryDirectory() as scratch, tarfile.open(options.out, "w:gz") as archive:
        for number_of_work, work in enumerate(works):
            path = os.path.join(options.pdf, work + ".pdf")
            if not os.path.exists(path):
                continue
            with fitz.open(path) as document:
                for kind in KINDS:
                    first = 0
                    for number, page in enumerate(document):
                        real = real_staffs(page)
                        if not real:
                            continue
                        image, staffs = page_image(
                            page, real, page_seed(work, number), kind == "photo"
                        )
                        image_path = os.path.join(scratch, "page.png")
                        cv2.imwrite(image_path, image)
                        try:
                            multi_staffs, preprocessed, debug, _ = detect_staffs_in_image(
                                image_path, config
                            )
                        except Exception as error:  # noqa: BLE001
                            eprint(f"{work} p{number + 1} {kind}: detection failed ({error})")
                            failed[kind] += 1
                            missed[kind] += len(real)
                            first += len(real)
                            continue
                        detected = [staff for multi in multi_staffs for staff in multi.staffs]
                        found = [(s.min_y, s.max_y, s.min_x, s.max_x) for s in detected]
                        owner, _ = pair(staffs, found)
                        regions = StaffRegions(multi_staffs)
                        for position, hits in owner.items():
                            name = f"{work}-{first + position:02d}"
                            if name not in labelled:
                                continue
                            if not hits:
                                missed[kind] += 1
                                continue
                            cut, _ = prepare_staff_image(
                                debug, position, detected[hits[0]], preprocessed, regions
                            )
                            data = cv2.imencode(".png", cut)[1].tobytes()
                            member = f"{FOLDER}/{kind}/{name}.png"
                            _add(archive, member, data)
                            index[kind].append(f"{member},datasets/SymbTr-work/{name}.tokens")
                            if options.samples and number_of_work < 3:
                                with open(
                                    os.path.join(options.samples, f"{kind}-{name}.png"), "wb"
                                ) as handle:
                                    handle.write(data)
                        first += len(real)
            eprint(f"  {number_of_work + 1}/{len(works)} {work}")

        for kind in KINDS:
            text = "\n".join(index[kind]) + "\n"
            _add(archive, f"{FOLDER}/index_{kind}.txt", text.encode("utf-8"))

    chosen = set(works)
    expected = sum(1 for name in labelled if STAFF.sub("", name) in chosen)
    eprint(f"\nlabelled staffs in these works: {expected}")
    for kind in KINDS:
        eprint(
            f"{kind:<6}: cut {len(index[kind])}, missed {missed[kind]},"
            f" pages where detection failed {failed[kind]}"
        )
    eprint(f"Wrote {options.out}")


if __name__ == "__main__":
    main()
