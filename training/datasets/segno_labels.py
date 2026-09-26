"""Give the printed corpus back the segnos its pictures show and its labels leave out.

Mus2 sets the segno as character 0x60 over the staff: above the first measure of a section, or above the
closing rule of a line. The converter never labelled it -- page_notes.py only notes that it "is a character
above the staff and no barline at all" -- so 2,852 segnos stand in the pages of 1,313 works while not one
label in the 50,353 printed training staffs holds a segno. The model was shown the sign thousands of times
with a label saying nothing is there, and learned to look past it: on handwritten pages it reads 3 segnos in
10, and four of its seven misses end the staff instead.

Each segno is found again in the pdf and given to the staff whose lines enclose its origin -- Mus2 places the
origin inside the staff and draws the sign upward from it (origin 123.3 on a staff spanning 113.4-134.4).
Its token then goes where the fifty hand-labelled segnos put theirs:

    at the head of the staff   after clef, key and usul, before a repeat start or the first note
    over a closing rule        after that rule, as the last token
    over a measure mid-staff   after the rule that opens that measure

The place in the label is found by counting the noteheads left of the sign on the page. Where the page's
noteheads and the label's notes differ in number the staff is left alone rather than guessed at.

Every picture drawn from the same Mus2 staff -- redrawn in other fonts, worn, given another usul, given
courtesy signs -- still shows the segno, so each of them gets the token too. Nothing is overwritten: the
changed labels go to a folder of their own and new index files point at them.

    python -m training.datasets.segno_labels
"""

import collections
import glob
import json
import os
import re

import fitz

from homr.simple_logging import eprint
from training.datasets.convert_symbtr import _staff_lines, dataset_root, git_root, symbtr_pdf
from training.datasets.courtesy import MARGIN
from training.datasets.page_notes import read_staff

SEGNO = 0x60
BASE = os.path.join(dataset_root, "SymbTr-work-v5")
OUT = os.path.join(dataset_root, "SymbTr-segno")
INDEX_DIR = os.path.join(dataset_root, "SymbTr-2.0.0")
# The pictures that are drawn from a Mus2 staff and so carry its segno. SymbTr-real holds scans of other
# publishers' pages, where a segno -- if there is one -- stands somewhere else.
FROM_MUS2 = ("SymbTr-work-v5", "SymbTr-worn", "SymbTr-fonts", "SymbTr-meters", "SymbTr-courtesy")

HEADER = ("clef", "keyAccidental", "timeSignature")
CLOSING = ("barline", "doublebarline", "bolddoublebarline", "repeatEnd", "repeatEndStart", "voltaStop",
           "dashedbarline")
OPENING = ("repeatStart", "voltaStart")
SEGNO_ROW = ["segno", ".", ".", ".", "."]
STAFF_KEY = re.compile(r"-\d{2}(?=-|$)")


def _bare(row: list[str]) -> str:
    return row[0].lstrip("?")


def _rows(path: str) -> list[list[str]]:
    return [line.split() for line in open(path, encoding="utf-8") if line.strip()]


def _notes(rows: list[list[str]]) -> int:
    return sum(1 for row in rows if _bare(row).startswith("note"))


def find_segnos() -> tuple[dict[str, dict], collections.Counter]:
    """Every staff of SymbTr-work-v5 with a segno on its page: {"<work>-NN": {"k": n, "yer": ...}}."""
    counts: collections.Counter = collections.Counter()
    places: dict[str, dict] = {}
    for path in sorted(glob.glob(os.path.join(symbtr_pdf, "*.pdf"))):
        work = os.path.splitext(os.path.basename(path))[0]
        try:
            document = fitz.open(path)
        except Exception:  # noqa: BLE001
            counts["pdf açılmadı"] += 1
            continue
        with document:
            staves = [(page.number, lines) for page in document for lines in _staff_lines(page)]
            for page in document:
                found = []
                for block in page.get_text("rawdict")["blocks"]:
                    for line in block.get("lines", []):
                        for span in line["spans"]:
                            if "Mus2" not in span.get("font", ""):
                                continue
                            for char in span.get("chars", []):
                                if ord(char["c"]) == SEGNO:
                                    x0, _, x1, _ = char["bbox"]
                                    found.append(((x0 + x1) / 2, char["origin"][1]))
                if not found:
                    continue
                numbered = [(n, lines) for n, (p, lines) in enumerate(staves) if p == page.number]
                for x, origin in found:
                    counts["senyo (pdf'te)"] += 1
                    distance = [(max(lines[0] - origin, origin - lines[-1], 0.0), n, lines)
                                for n, lines in numbered]
                    if not distance:
                        counts["sayfada porte yok"] += 1
                        continue
                    gap, number, lines = min(distance)
                    if gap > MARGIN:
                        counts["kesimin dışında"] += 1
                        continue
                    name = f"{work}-{number:02d}"
                    tokens = os.path.join(BASE, name + ".tokens")
                    if not os.path.exists(tokens):
                        counts["porte veride yok"] += 1
                        continue
                    heads = read_staff(page, lines)
                    if len(heads) != _notes(_rows(tokens)):
                        counts["nota sayısı tutmuyor, bırakıldı"] += 1
                        continue
                    k = sum(1 for head in heads if head["x"] < x)
                    spot = "bas" if k == 0 else "son" if k == len(heads) else "orta"
                    if name in places:
                        counts["aynı portede ikinci senyo, bırakıldı"] += 1
                        continue
                    places[name] = {"k": k, "yer": spot}
                    counts[f"yerleşti: {spot}"] += 1
    return places, counts


def segno_at(rows: list[list[str]], k: int, spot: str) -> int | None:
    """Where the segno row goes in this label, or None if the label leaves no sound place for it."""
    notes = [i for i, row in enumerate(rows) if _bare(row).startswith("note")]
    if spot == "son":
        return len(rows)
    if spot == "bas":
        at = 0
        while at < len(rows) and _bare(rows[at]).startswith(HEADER):
            at += 1
        return at
    if not 0 < k < len(notes):
        return None
    left, right = notes[k - 1], notes[k]
    closing = [i for i in range(left + 1, right) if _bare(rows[i]) in CLOSING]
    if closing:
        return closing[-1] + 1
    opening = [i for i in range(left + 1, right) if _bare(rows[i]) in OPENING]
    return opening[0] if opening else right


def staff_key(image: str, known: set[str]) -> str | None:
    """The "<work>-NN" a picture was drawn from, whatever was added to its name after that."""
    name = os.path.splitext(os.path.basename(image))[0]
    for match in STAFF_KEY.finditer(name):
        key = name[: match.end()]
        if key in known:
            return key
    return None


# The exams drawn from the same Mus2 staffs. Their pictures show the segno too, so an exam left on the old
# labels would mark a model wrong for reading a sign that is on the page. Corrected copies go under
# SymbTr-segno/exams/, keeping each exam's own folder and file name, so they can be laid over the old ones.
EXAMS = ("SymbTr-courtesy/index_test.txt", "SymbTr-fonts/index_test.txt", "SymbTr-fonts/index_test_unseen.txt",
         "SymbTr-meters/index_test.txt", "SymbTr-meters/index_test_fonts.txt", "SymbTr-worn/index_test.txt")


class _Relabeller:
    def __init__(self, places: dict[str, dict]) -> None:
        self.places = places
        self.known = {os.path.splitext(os.path.basename(p))[0]
                      for p in glob.glob(os.path.join(BASE, "*.tokens"))}
        self.base_notes = {name: _notes(_rows(os.path.join(BASE, name + ".tokens"))) for name in places}
        self.written: dict[str, str] = {}
        self.counts: collections.Counter = collections.Counter()

    def line(self, line: str, split: str) -> str:
        image, tokens = line.strip().split(",")
        folder = image.split("/")[1] if image.count("/") >= 2 else ""
        key = staff_key(image, self.known) if folder in FROM_MUS2 else None
        if key is None or key not in self.places:
            return line.strip() + "\n"
        if tokens not in self.written:
            self.written[tokens] = self._rewrite(tokens, key, f"{folder} ({split})")
        return f"{image},{self.written[tokens]}\n"

    def _rewrite(self, tokens: str, key: str, where: str) -> str:
        rows = _rows(os.path.join(git_root, tokens))
        if _notes(rows) != self.base_notes[key]:
            self.counts["türevde nota sayısı farklı, bırakıldı"] += 1
            return tokens
        if any(_bare(row) == "segno" for row in rows):
            self.counts["zaten senyolu"] += 1
            return tokens
        at = segno_at(rows, self.places[key]["k"], self.places[key]["yer"])
        if at is None:
            self.counts["yer bulunamadı, bırakıldı"] += 1
            return tokens
        rows = rows[:at] + [list(SEGNO_ROW)] + rows[at:]
        target = os.path.join(OUT, os.path.relpath(os.path.join(git_root, tokens), dataset_root))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as handle:
            handle.writelines(" ".join(row) + "\n" for row in rows)
        self.counts[f"düzeltildi: {where}"] += 1
        return os.path.relpath(target, git_root).replace(os.sep, "/")

    def index(self, source: str, target: str, split: str) -> list[str]:
        lines_out = [self.line(line, split) for line in open(source, encoding="utf-8") if line.strip()]
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as handle:
            handle.writelines(lines_out)
        name = os.path.relpath(target, OUT)
        self.counts[f"satır: {name}"] = len(lines_out)
        self.counts[f"senyolu satır: {name}"] = sum(1 for l in lines_out if "SymbTr-segno/" in l)
        return lines_out


def relabel(places: dict[str, dict]) -> collections.Counter:
    work = _Relabeller(places)
    for split in ("train", "val", "test"):
        lines_out = work.index(os.path.join(INDEX_DIR, f"index_{split}.txt"),
                               os.path.join(OUT, f"index_{split}.txt"), split)
        if split == "test":
            # The exam of the segno itself: every printed test staff that has one.
            with open(os.path.join(OUT, "index_test_senyolu.txt"), "w", encoding="utf-8") as handle:
                handle.writelines(l for l in lines_out if "SymbTr-segno/" in l)
    for exam in EXAMS:
        source = os.path.join(dataset_root, exam)
        if os.path.exists(source):
            work.index(source, os.path.join(OUT, "exams", exam), "sınav")
    return work.counts


def main() -> None:
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--find", action="store_true",
                        help="Read the pdfs again even if senyo_yerleri.json is already there.")
    options = parser.parse_args()
    cached = os.path.join(OUT, "senyo_yerleri.json")
    if os.path.exists(cached) and not options.find:
        places = json.load(open(cached, encoding="utf-8"))
        eprint(f"{len(places)} segno places read from {cached}")
    else:
        places, found = find_segnos()
        for key, n in sorted(found.items(), key=lambda kv: -kv[1]):
            eprint(f"   {n:6d}  {key}")
        os.makedirs(OUT, exist_ok=True)
        with open(cached, "w", encoding="utf-8") as handle:
            json.dump(places, handle, indent=0)
    done = relabel(places)
    for key, n in sorted(done.items()):
        eprint(f"   {n:6d}  {key}")


if __name__ == "__main__":
    main()
