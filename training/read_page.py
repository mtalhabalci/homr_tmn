"""Read whole score pages the way a photograph would be read, and write MusicXML.

Everything else in training/ was measured on SymbTr's own Mus2 pages. This is
for scores from anywhere: a pdf, scanned or typeset, is rendered page by page
to an image and nothing else about it is used. homr finds and cuts the
staffs, the fine-tuned model reads each one, the makam resolver works out what
sounds, and the result is written as MusicXML beside a picture of every staff
with what was read under it.

When the same piece is in SymbTr, the reading can also be set against its
.mu2: notes are lined up by edit distance and counted, pitch with its
accidental and then with its duration. Two editions of a piece are seldom
identical, so this is a lower bound on how well the page was read.

    python -m training.read_page --folder "G:/.../neyzen" --checkpoint model.pth --out "G:/.../neyzen_okuma"
"""

import argparse
import json
import os
import tempfile
from concurrent.futures import Future
from fractions import Fraction

import cv2
import fitz
import numpy as np
import torch

from homr import main as homr_main
from homr.main import ProcessingConfig, detect_staffs_in_image
from homr.makam_key import commas_for_lift, resolve_sounding
from homr.music_xml_generator import XmlGeneratorArguments, generate_xml
from homr.simple_logging import eprint
from homr.staff_parsing import prepare_staff_image
from homr.staff_regions import StaffRegions
from homr.transformer.configs import Config
from homr.transformer.vocabulary import EncodedSymbol
from training.architecture.transformer.tromr_arch import load_model
from training.datasets.convert_symbtr import NEAREST_AEU, TURKISH_NOTE_NAMES, parse_note_name, read_mu2
from training.evaluate_makam import _align
from training.transformer.image_utils import ndarray_to_tensor, pad_to_3_dims

PAGE_WIDTH = 1920
LETTER_NAMES = {letter: name.capitalize() for name, letter in TURKISH_NOTE_NAMES.items()}


def _no_title(*_: object) -> Future:
    done: Future = Future()
    done.set_result("")
    return done


def named(pitch: str, commas: int) -> str:
    name = f"{LETTER_NAMES.get(pitch[0], pitch[0])}{pitch[1:]}"
    return name + (f"#{commas}" if commas > 0 else f"b{-commas}" if commas < 0 else "")


def read_pages(path: str, model: torch.nn.Module, scratch: str) -> list[dict]:
    """Every staff of every page: the cut homr made, and the symbols read from it."""
    config = ProcessingConfig(False, False, False, False, -1, False)
    device = next(model.parameters()).device
    staffs = []
    with fitz.open(path) as document:
        for page in document:
            zoom = PAGE_WIDTH / page.rect.width
            image_path = os.path.join(scratch, "page.png")
            page.get_pixmap(matrix=fitz.Matrix(zoom, zoom)).save(image_path)
            try:
                multi_staffs, preprocessed, debug, _ = detect_staffs_in_image(image_path, config)
            except Exception as error:  # noqa: BLE001
                eprint(f"  page {page.number + 1}: no staffs found ({error})")
                continue
            regions = StaffRegions(multi_staffs)
            found = sorted((s for multi in multi_staffs for s in multi.staffs), key=lambda s: s.min_y)
            for number, staff in enumerate(found):
                cut, _ = prepare_staff_image(debug, number, staff, preprocessed, regions)
                gray = cut if cut.ndim == 2 else cv2.cvtColor(cut, cv2.COLOR_BGR2GRAY)
                tensor = pad_to_3_dims(ndarray_to_tensor(gray)).to(device)
                with torch.no_grad():
                    symbols = model.generate(tensor.unsqueeze(0))
                staffs.append({
                    "page": page.number + 1,
                    "staff": number + 1,
                    "cut": gray,
                    "symbols": [s for s in symbols if not s.is_control_symbol()],
                })
    return staffs


def _value(rhythm: str) -> Fraction:
    kern = rhythm.split("_", 1)[1].rstrip("G")
    dots = kern.count(".")
    base = kern.rstrip(".")
    return Fraction(1, int(base)) * (2 - Fraction(1, 2**dots)) if base.isdigit() and int(base) else Fraction(0)


def sounding_notes(voice: list[EncodedSymbol]) -> list[tuple[str, int, Fraction]]:
    """(pitch, commas, duration) of every note as it sounds, tied halves joined."""
    notes: list[tuple[str, int, Fraction]] = []
    for symbol in resolve_sounding(voice):
        if not symbol.rhythm.startswith("note") or symbol.rhythm.endswith("G"):
            continue
        commas = commas_for_lift(symbol.sounding) if symbol.sounding else 0
        duration = _value(symbol.rhythm)
        if "tieStop" in symbol.articulation and notes and notes[-1][0] == symbol.pitch:
            pitch, held, length = notes[-1]
            notes[-1] = (pitch, held, length + duration)
            continue
        notes.append((symbol.pitch, commas or 0, duration))
    return notes


def mu2_notes(path: str) -> list[tuple[str, int, Fraction]]:
    notes = []
    for event in read_mu2(path).events:
        parsed = parse_note_name(event["name"])
        if parsed and not event["grace"]:
            notes.append((parsed[0], parsed[1], event["duration"]))
    return notes


def compare(read: list[tuple[str, int, Fraction]], reference: list[tuple[str, int, Fraction]]) -> dict:
    """Line the two up by pitch and accidental, then count."""
    def key(note: tuple[str, int, Fraction]) -> tuple[str, int]:
        return note[0], NEAREST_AEU.get(note[1], note[1])

    pairs = _align([key(n) for n in reference], [key(n) for n in read])
    matched = sum(1 for a, b in pairs if a is not None and a == b)
    altered = [(a, b) for a, b in pairs if a is not None and a[1] != 0]
    altered_ok = sum(1 for a, b in altered if a == b)
    return {
        "reference notes": len(reference),
        "notes read": len(read),
        "pitch and accidental right": f"{matched}/{len(reference)} = {100 * matched / max(len(reference), 1):.1f}%",
        "altered notes right": f"{altered_ok}/{len(altered)} = {100 * altered_ok / max(len(altered), 1):.1f}%",
        "missed": sum(1 for a, b in pairs if b is None),
        "extra": sum(1 for a, b in pairs if a is None),
    }


def text_of(symbols: list[EncodedSymbol]) -> str:
    """What was read, the way a musician would say it."""
    words = []
    for s in resolve_sounding(symbols):
        if s.rhythm.startswith("note"):
            commas = commas_for_lift(s.sounding) if s.sounding else 0
            words.append(named(s.pitch, commas or 0) + ("~" if "tieStart" in s.articulation else ""))
        elif s.rhythm.startswith("rest"):
            words.append("·")
        elif s.rhythm == "keyAccidental":
            words.append(f"[{named(s.pitch, commas_for_lift(s.lift) or 0)}]")
        elif s.rhythm.startswith("timeSignature_"):
            words.append(f"({s.rhythm[14:]})")
        elif s.rhythm in ("barline", "repeatEnd", "repeatStart", "bolddoublebarline", "voltaStop"):
            words.append({"barline": "|", "repeatEnd": ":|", "repeatStart": "|:", "bolddoublebarline": "||",
                          "voltaStop": "|"}[s.rhythm])
    return " ".join(words)


def main() -> None:  # noqa: PLR0915
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", required=True, help="Folder of pdfs to read.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--match", action="append", default=[],
                        help="pdf-stem=mu2-path, to compare a reading with the same piece in SymbTr.")
    parser.add_argument("--tokens", action="store_true",
                        help="Also write what was read from each staff as a .tokens file beside its picture, "
                             "in the training data's format, to build labels on.")
    options = parser.parse_args()
    matches = dict(item.split("=", 1) for item in options.match)

    homr_main.detect_title = _no_title
    config = Config()
    config.filepaths.checkpoint = options.checkpoint
    model = load_model(config)
    model.eval_mode()
    os.makedirs(options.out, exist_ok=True)
    summary = {}
    with tempfile.TemporaryDirectory() as scratch:
        for name in sorted(os.listdir(options.folder)):
            if not name.lower().endswith(".pdf"):
                continue
            stem = name[:-4]
            eprint(f"Reading {stem}")
            staffs = read_pages(os.path.join(options.folder, name), model, scratch)
            voice: list[EncodedSymbol] = []
            for staff in staffs:
                voice += staff["symbols"] + [EncodedSymbol("newline")]
            # One malformed symbol must not cost the other scores their reading.
            try:
                xml = generate_xml(XmlGeneratorArguments(), [voice], stem)
                xml_path = os.path.join(scratch, "out.musicxml")
                xml.write(xml_path)
                with open(xml_path, "rb") as source, open(os.path.join(options.out, stem + ".musicxml"), "wb") as target:
                    target.write(source.read())
            except Exception as error:  # noqa: BLE001
                eprint(f"  no MusicXML for {stem}: {error}")
            folder = os.path.join(options.out, stem)
            os.makedirs(folder, exist_ok=True)
            record = []
            for staff in staffs:
                png = f"p{staff['page']}-s{staff['staff']:02d}.png"
                with open(os.path.join(folder, png), "wb") as handle:
                    handle.write(cv2.imencode(".png", staff["cut"])[1].tobytes())
                if options.tokens:
                    with open(os.path.join(folder, png[:-4] + ".tokens"), "w", encoding="utf-8") as handle:
                        handle.writelines(
                            f"{s.rhythm} {s.pitch} {s.lift} {s.articulation} {s.position}\n" for s in staff["symbols"]
                        )
                record.append({"page": staff["page"], "staff": staff["staff"], "image": f"{stem}/{png}",
                               "read": text_of(staff["symbols"])})
            result = {"staffs": len(staffs), "symbols": sum(len(s["symbols"]) for s in staffs)}
            if stem in matches:
                result.update(compare(sounding_notes(voice), mu2_notes(matches[stem])))
                result["compared with"] = os.path.basename(matches[stem])
            summary[stem] = result
            with open(os.path.join(options.out, stem + ".json"), "w", encoding="utf-8") as handle:
                json.dump({"summary": result, "staffs": record}, handle, ensure_ascii=False, indent=1)
            eprint(f"  {result}")
    with open(os.path.join(options.out, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
