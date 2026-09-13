"""Where the SymbTr pdfs and their .mu2 scores disagree, for the corpus maintainers.

While building training data from SymbTr every staff of every pdf was read
back from the page -- Mus2 draws notation as text, so each notehead, sign and
digit can be located -- and compared with the .mu2 it was engraved from. Most
agree. Some pdfs were evidently engraved from a slightly different version of
the piece. This writes those disagreements out, one file per kind:

    notes.csv           a note whose pitch, as the pdf makes it sound, is not
                        the .mu2's. Sounding pitch, not the printed sign: a
                        courtesy sign that restates the key signature is not a
                        disagreement, and neither is a 2- or 3-comma step
                        printed with the nearest AEU sign, the rounding many
                        scores apply throughout.
    key_signatures.csv  the key signature printed on the first staff against
                        the .mu2's key signature row.
    time_signatures.csv the time signature printed on the first staff against
                        the .mu2 header.
    staffs_to_check.csv staffs whose notes the pdf prints differently from the
                        .mu2 (different count or pitches). Unverified.
    not_compared.csv    works that could not be compared, and why.

    python -m training.symbtr_report --out "G:/.../homr_makam/symbtr_rapor"
"""

import argparse
import collections
import csv
import os
import re
import tempfile
from pathlib import Path

from homr.simple_logging import eprint
from training.datasets import convert_symbtr as cs


def write(path: str, rows: list[dict]) -> None:
    if not rows:
        with open(path, "w", encoding="utf-8-sig", newline="") as handle:
            handle.write("(none)\n")
        return
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def musicxml_blanks(folder: str) -> tuple[int, int, int]:
    """MusicXML files with an accidental element left empty, and how many."""
    files = with_blank = blanks = 0
    for path in Path(folder).glob("*.xml"):
        files += 1
        text = path.read_text(encoding="utf-8", errors="replace")
        count = len(re.findall(r"<accidental>\s*</accidental>|<key-accidental>\s*</key-accidental>", text))
        if count:
            with_blank += 1
            blanks += count
    return files, with_blank, blanks


def main() -> None:  # noqa: PLR0915
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Folder to write the report into.")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N works.")
    options = parser.parse_args()
    os.makedirs(options.out, exist_ok=True)

    cs.comparison = {"notes": [], "keys": [], "times": [], "staffs": []}
    stems = sorted(path.stem for path in Path(cs.symbtr_pdf).glob("*.pdf"))[: options.limit]
    mu2_index = cs.build_mu2_index()
    unresolved: collections.Counter = collections.Counter()
    skipped = []
    compared = 0
    with tempfile.TemporaryDirectory() as scratch:
        cs.working_dir = scratch
        for number, stem in enumerate(stems, start=1):
            try:
                cs.convert_work(stem, True, unresolved, mu2_index.get(stem))
                compared += 1
            except cs.SkippedWork as reason:
                skipped.append({"work": stem, "reason": str(reason)})
            except Exception as error:  # noqa: BLE001
                skipped.append({"work": stem, "reason": f"{type(error).__name__}: {error}"})
            if number % 200 == 0:
                eprint(f"  {number}/{len(stems)}")

    found = cs.comparison
    write(os.path.join(options.out, "notes.csv"), found["notes"])
    write(os.path.join(options.out, "key_signatures.csv"), found["keys"])
    write(os.path.join(options.out, "time_signatures.csv"), found["times"])
    write(os.path.join(options.out, "staffs_to_check.csv"), found["staffs"])
    write(os.path.join(options.out, "not_compared.csv"), skipped)

    by_cause = collections.Counter(row["cause"].split(" (")[0] for row in found["notes"])
    works_with_notes = len({row["work"] for row in found["notes"]})
    files, with_blank, blanks = musicxml_blanks(os.path.join(cs.symbtr_root, "MusicXML"))
    reasons = collections.Counter(re.sub(r"\d+", "N", row["reason"]) for row in skipped)
    summary = {
        "pdfs": len(stems), "compared": compared, "not compared": len(skipped),
        "notes that sound different": len(found["notes"]), "in works": works_with_notes,
        "by cause": dict(by_cause),
        "key signatures that differ": len(found["keys"]),
        "time signatures that differ": len(found["times"]),
        "staffs to check": len(found["staffs"]),
        "musicxml files": files, "musicxml files with an empty accidental": with_blank,
        "empty accidental elements": blanks,
        "not compared, by reason": dict(reasons.most_common()),
    }
    with open(os.path.join(options.out, "summary.txt"), "w", encoding="utf-8") as handle:
        for key, value in summary.items():
            handle.write(f"{key}: {value}\n")
    for key, value in summary.items():
        eprint(f"{key}: {value}")


if __name__ == "__main__":
    main()
