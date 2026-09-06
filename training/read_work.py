"""Have the model read one work and write down exactly what it saw.

The class-by-class numbers say how often the model is right; they do not show
what being right or wrong looks like on a page. This cuts a work's staffs out of
its pdf, lets the model read each one on its own, and records every symbol it
produced along with the attention coordinates that place it roughly under the
staff. The .mu2 note list goes in beside it as the reference.

    python -m training.read_work --work rast--medhal--hafif--... --out /tmp/demo

Writes <out>/<work>.json plus one png per staff. The comparison and the page are
built from that file afterwards; this side only gathers facts.

Works the dataset skipped can be read too. Staff cutting does not care whether
the measures close on the usul, and neither does the model.
"""

import argparse
import json
import os
import sys

import torch

from homr.simple_logging import eprint
from homr.staff_parsing import add_image_into_tr_omr_canvas
from homr.transformer.configs import Config
from training.architecture.transformer.tromr_arch import load_model
from training.datasets.convert_symbtr import (
    _barlines,
    _mus2_glyphs,
    _staff_lines,
    parse_note_name,
    printed_key_signature,
    read_mu2,
    symbtr_mu2,
    symbtr_pdf,
    target_page_width,
)
from training.transformer.image_utils import (
    ndarray_to_tensor,
    pad_to_3_dims,
    prepare_for_tensor,
    read_image_to_ndarray,
)


def cut_staffs(stem: str, out_dir: str) -> list[dict]:
    """Render one png per staff, the same crop the training data uses."""
    import fitz  # noqa: PLC0415

    path = os.path.join(symbtr_pdf, stem + ".pdf")
    if not os.path.exists(path):
        eprint(f"No pdf at {path}")
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)
    staffs = []
    with fitz.open(path) as document:
        for page in document:
            glyphs = _mus2_glyphs(page)
            for lines in _staff_lines(page):
                staffs.append(
                    {
                        "page": page.number,
                        "lines": lines,
                        "bars": len(_barlines(page, lines, glyphs)),
                    }
                )
        for index, staff in enumerate(staffs):
            page = document[staff["page"]]
            clip = fitz.Rect(
                0, staff["lines"][0] - 34, page.rect.width, staff["lines"][-1] + 34
            )
            dpi = round(target_page_width / page.rect.width * 72)
            name = f"{stem}-{index:02d}.png"
            page.get_pixmap(clip=clip, dpi=dpi).save(os.path.join(out_dir, name))
            staff["image"] = name
            del staff["lines"]
    return staffs


def reference_notes(stem: str) -> dict:
    """The .mu2 note list, which is what the engraver worked from."""
    path = os.path.join(symbtr_mu2, stem + ".mu2")
    if not os.path.exists(path):
        return {}
    score = read_mu2(path)
    notes = []
    for event in score.events:
        parsed = parse_note_name(event["name"])
        if parsed is None:
            continue
        pitch, commas = parsed
        notes.append(
            {
                "pitch": pitch,
                "commas": commas,
                "duration": str(event["duration"]),
                "grace": bool(event.get("grace")),
            }
        )
    printed = score.key
    try:
        import fitz  # noqa: PLC0415

        with fitz.open(os.path.join(symbtr_pdf, stem + ".pdf")) as document:
            printed = printed_key_signature(document, score.key)
    except Exception as error:  # noqa: BLE001
        eprint(f"Could not read the printed key signature: {error}")
    return {
        "usul": f"{score.numerator}/{score.denominator}",
        "key_mu2": score.key,
        "key_printed": printed,
        "notes": notes,
    }


def read_with_model(images: list[str], out_dir: str, checkpoint: str | None) -> list[list[dict]]:
    config = Config()
    if checkpoint:
        config.filepaths.checkpoint = checkpoint
    if not os.path.exists(config.filepaths.checkpoint):
        eprint(f"No checkpoint at {config.filepaths.checkpoint}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config)
    model.eval_mode()

    readings = []
    for name in images:
        image = read_image_to_ndarray(os.path.join(out_dir, name))
        # No distortion here. The evaluation feeds the model roughened images on
        # purpose, to stand in for a photograph; a demonstration should show what
        # it reads off the page as printed.
        image = add_image_into_tr_omr_canvas(image)
        tensor = pad_to_3_dims(ndarray_to_tensor(prepare_for_tensor(image))).to(device)
        with torch.no_grad():
            symbols = model.generate(tensor.unsqueeze(0) if tensor.dim() == 3 else tensor)
        readings.append(
            [
                {
                    "rhythm": symbol.rhythm,
                    "pitch": symbol.pitch,
                    "lift": symbol.lift,
                    "articulation": symbol.articulation,
                    "position": symbol.position,
                    "at": symbol.coordinates,
                }
                for symbol in symbols
                if not symbol.is_control_symbol()
            ]
        )
        eprint(f"  {name}: {len(readings[-1])} symbols")
    return readings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, help="Work stem, without .pdf")
    parser.add_argument("--checkpoint", default=None, help="Model to read with.")
    parser.add_argument("--out", default="demo", help="Where to write the png and json.")
    options = parser.parse_args()

    stem = options.work.removesuffix(".pdf")
    eprint(f"Cutting {stem}")
    staffs = cut_staffs(stem, options.out)
    eprint(f"  {len(staffs)} staffs, {sum(s['bars'] for s in staffs)} barlines")

    readings = read_with_model([s["image"] for s in staffs], options.out, options.checkpoint)
    for staff, reading in zip(staffs, readings):
        staff["symbols"] = reading

    result = {"work": stem, "staffs": staffs, "reference": reference_notes(stem)}
    destination = os.path.join(options.out, stem + ".json")
    with open(destination, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=1)
    eprint(f"Wrote {destination}")


if __name__ == "__main__":
    main()
