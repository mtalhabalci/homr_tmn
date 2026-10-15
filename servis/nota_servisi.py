"""Photo or PDF of a makam score in, MusicXML out: the HTTP service the Musiki Akademisi app calls.

    python -m servis.nota_servisi                 listens on 127.0.0.1:OMR_PORT (default 8765)

    POST /v1/omr            multipart: dosya (JPEG, PNG, WebP or PDF, at most 20 MB), tur (basili | el_yazisi,
                            optional; one model reads both, so it is only echoed back), sayfalar (optional, PDF
                            only: 1-based pages such as "1-3,5"; a bad range is 400 gecersiz_sayfalar)
                            ?kuyruk=1 queues the job and answers {"isId", "sayfaSayisi", "toplamSayfa"} at once;
                            jobs live in memory only, so a restart loses them (GET answers 404, the app sends again)
                            Besides the MusicXML the answer carries "porteler" (per staff: page, box on the page image,
                            first and last measure) and "sayfaResimleri" (per page: image size and DPI)
    GET  /v1/omr/{isId}     {"durum": "isleniyor" | "tamam" | "hata" | "iptal", "islenenSayfa", "islenenPorte", ...}
    DELETE /v1/omr/{isId}   a running job stops before its next staff and stays as durum "iptal" (202); a finished
                            one is removed (200)
    POST /v1/sayfa-sayisi   multipart dosya -> {"sayfaSayisi"}
    GET  /v1/saglik         {"modelVersion": ...}

When OMR_ANAHTAR is set, the X-Omr-Anahtar header must equal it.

The reading is the thesis's final reader: the fine-tuned staff model (v20) with its decoding rules, the makam
resolver and the MusicXML writer. Staffs are found and cut as training/read_page.py does for the page exams.
The app is not to show which open-source project the reader grew from, so the file's <software> is OMR_YAZILIM
and no response names it.

Environment: OMR_MODEL (weights), OMR_MODEL_SURUM (reported version), OMR_YAZILIM (<software>), OMR_ANAHTAR,
OMR_PORT, OMR_GPU_SEGMENTASYON=1 to run staff finding on the GPU.
"""

import os
import re
import tempfile
import threading
import time
import uuid
import xml.etree.ElementTree as ET
from concurrent.futures import Future
from fractions import Fraction

import cv2
import fitz
import torch
from fastapi import FastAPI, File, Form, Header, UploadFile
from fastapi.responses import JSONResponse

from homr import main as reader_main
from homr.autocrop import autocrop_box
from homr.main import ProcessingConfig, detect_staffs_in_image
from homr.music_xml_generator import XmlGeneratorArguments, generate_xml
from homr.staff_parsing import prepare_staff_image
from homr.staff_regions import StaffRegions
from homr.tempo_reader import read_tempo_marks
from homr.transformer.configs import Config
from homr.transformer.vocabulary import EncodedSymbol
from training.architecture.transformer.tromr_arch import load_model
from training.transformer.image_utils import ndarray_to_tensor, pad_to_3_dims

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.environ.get("OMR_MODEL", os.path.join(
    ROOT, "training", "architecture", "transformer", "pytorch_model_1-111a84bed30c8c6a373ad135e05633ccd8c63b11.pth"))
MODEL_VERSION = os.environ.get("OMR_MODEL_SURUM", "v20+kurallar-2026-10-10")
SOFTWARE = os.environ.get("OMR_YAZILIM", "Musiki Akademisi OMR")
KEY = os.environ.get("OMR_ANAHTAR", "")
MAX_BYTES = 20 * 1024 * 1024
PAGE_WIDTH = 1920  # a pdf page is rendered at this width, as for the page exams
LOW_CONFIDENCE = 0.5
TYPES = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp", "application/pdf": ".pdf"}


def _no_title(*_: object) -> Future:
    """The title is not read (it would need OCR); the app keeps its own."""
    done: Future = Future()
    done.set_result("")
    return done


def page_count(path: str) -> int:
    if not path.endswith(".pdf"):
        return 1
    with fitz.open(path) as document:
        return document.page_count


def measures_off_meter(musicxml: str) -> list[int]:
    """Measures whose notes and rests do not add up to the time signature: a duration was misread. The first and
    last measure may be short (an upbeat and the measure that completes it), so they count only when too long."""
    root = ET.fromstring(musicxml)
    measures = root.findall("./part/measure")
    divisions, meter, off = 1, None, []
    for index, measure in enumerate(measures):
        for attributes in measure.findall("attributes"):
            divisions = int(attributes.findtext("divisions") or divisions)
            time_signature = attributes.find("time")
            if time_signature is not None:
                try:
                    meter = Fraction(int(time_signature.findtext("beats")), int(time_signature.findtext("beat-type")))
                except (TypeError, ValueError):
                    meter = None
        if meter is None or measure.find("attributes/measure-style/multiple-rest") is not None:
            continue
        voices: dict[str, int] = {}
        for note in measure.findall("note"):
            if note.find("chord") is None and note.find("grace") is None:
                voice = note.findtext("voice") or "1"
                voices[voice] = voices.get(voice, 0) + int(note.findtext("duration") or 0)
        length = Fraction(max(voices.values(), default=0), divisions * 4)
        edge = index in (0, len(measures) - 1)
        if length > meter or (length < meter and not edge):
            off.append(int(measure.get("number")))
    return off


def tempo_measures(found: list, staff_boxes: list[tuple[int, list[int]]],
                   staff_measures: list[tuple[int, int]]) -> list[dict]:
    """The measure each tempo mark starts: a mark near the start of its staff opens the staff's first measure,
    one further along the measure under it, counting the staff's measures as equally wide. One mark per measure,
    the surest."""
    chosen: dict[int, dict] = {}
    for staff, mark in found:
        if staff >= len(staff_measures):
            continue
        first, last = staff_measures[staff]
        if first > last:
            continue
        x0, _, x1, _ = staff_boxes[staff][1]
        along = (mark.box[0] - x0) / max(1, x1 - x0)
        count = last - first + 1
        measure = first if along < 0.2 else first + min(count - 1, int(along * count))
        entry = {"olcu": measure, "birim": mark.unit, "noktali": mark.dotted, "deger": mark.per_minute,
                 "guven": mark.confidence, "porte": staff + 1, "sayfa": staff_boxes[staff][0],
                 "kutu": [int(v) for v in mark.box]}
        if measure not in chosen or chosen[measure]["guven"] < mark.confidence:
            chosen[measure] = entry
    return [chosen[m] for m in sorted(chosen)]


class ReadError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


class Cancelled(Exception):
    """A queued reading stopped by DELETE /v1/omr/{isId}."""


def parse_pages(spec: str | None, total: int) -> list[int]:
    """'1-3,5' -> [1, 2, 3, 5]: 1-based, sorted, each page once. Empty means every page."""
    if not spec or not spec.strip():
        return list(range(1, total + 1))
    chosen: set[int] = set()
    for part in spec.replace(" ", "").split(","):
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        first, last = (int(match.group(1)), int(match.group(2) or match.group(1))) if match else (0, 0)
        if not match or first < 1 or last < first or last > total:
            raise ReadError(400, "gecersiz_sayfalar",
                            f"Geçersiz sayfa aralığı: '{part}'. Biçim '1-3,5' gibi olmalı; dosyada {total} sayfa var.")
        chosen.update(range(first, last + 1))
    return sorted(chosen)


class Reader:
    """The model, loaded once; one reading at a time, since the GPU holds a single copy."""

    def __init__(self) -> None:
        reader_main.detect_title = _no_title
        config = Config()
        config.filepaths.checkpoint = MODEL_PATH
        self.model = load_model(config)
        self.model.eval_mode()
        self.device = next(self.model.parameters()).device
        self.lock = threading.Lock()
        self.processing = ProcessingConfig(False, False, False, False, -1,
                                           os.environ.get("OMR_GPU_SEGMENTASYON") == "1")

    def _page_image(self, path: str, scratch: str, page_no: int) -> tuple[str, float | None]:
        """The page as an image file, and its resolution in DPI (a PDF page is rendered; a photo has none)."""
        if not path.endswith(".pdf"):
            return path, None
        with fitz.open(path) as document:
            page = document[page_no - 1]
            zoom = PAGE_WIDTH / page.rect.width
            image = os.path.join(scratch, f"sayfa-{page_no}.png")
            page.get_pixmap(matrix=fitz.Matrix(zoom, zoom)).save(image)
        return image, round(72 * zoom, 2)

    def read(self, path: str, pages: list[int] | None = None, progress=None,
             cancel: threading.Event | None = None) -> dict:
        """Read the given 1-based pages of a PDF (all when None; an image is page 1). progress(pages_done, staffs)
        is called after each page; cancel, when set, stops the reading before the next staff."""
        start = time.perf_counter()
        warnings: list[str] = []
        voice: list[EncodedSymbol] = []
        staffs = 0
        page_info: list[dict] = []
        staff_boxes: list[tuple[int, list[int]]] = []
        tempo_found: list[tuple[int, object]] = []  # (staff number from 0, TempoMark)
        total = page_count(path)
        pages = pages or list(range(1, total + 1))
        with self.lock, tempfile.TemporaryDirectory() as scratch:
            for done, page_no in enumerate(pages):
                if cancel is not None and cancel.is_set():
                    raise Cancelled
                image, dpi = self._page_image(path, scratch, page_no)
                page_pixels = cv2.imread(image)
                where = f"{page_no}. sayfa" if total > 1 else "Sayfa"
                try:
                    multi_staffs, preprocessed, debug, _ = detect_staffs_in_image(image, self.processing)
                except Exception:  # noqa: BLE001
                    warnings.append(f"{where}: porte bulunamadı.")
                    continue
                regions = StaffRegions(multi_staffs)
                # staffs are found on the page cropped to the paper and scaled; their boxes go back to the image
                height, width = page_pixels.shape[:2]
                box = autocrop_box(page_pixels) or (0, 0, width, height)
                scale_x, scale_y = box[2] / preprocessed.shape[1], box[3] / preprocessed.shape[0]
                page_info.append({"sayfa": page_no, "genislik": width, "yukseklik": height, "dpi": dpi})
                found = sorted((s for multi in multi_staffs for s in multi.staffs), key=lambda s: s.min_y)
                page_first = len(staff_boxes)
                for number, staff in enumerate(found):
                    if cancel is not None and cancel.is_set():
                        raise Cancelled
                    try:
                        cut, _ = prepare_staff_image(debug, number, staff, preprocessed, regions)
                    except cv2.error:
                        warnings.append(f"{where}, {number + 1}. porte kesilemedi; atlandı.")
                        continue
                    gray = cut if cut.ndim == 2 else cv2.cvtColor(cut, cv2.COLOR_BGR2GRAY)
                    tensor = pad_to_3_dims(ndarray_to_tensor(gray)).to(self.device)
                    with torch.no_grad():
                        symbols = self.model.generate(tensor.unsqueeze(0))
                    voice += [s for s in symbols if not s.is_control_symbol()] + [EncodedSymbol("newline")]
                    staff_boxes.append((page_no, [round(box[0] + staff.min_x * scale_x),
                                                  round(box[1] + staff.min_y * scale_y),
                                                  round(box[0] + staff.max_x * scale_x),
                                                  round(box[1] + staff.max_y * scale_y)]))
                    staffs += 1
                gray_page = cv2.cvtColor(page_pixels, cv2.COLOR_BGR2GRAY)
                for mark in read_tempo_marks(gray_page, [tuple(b) for _, b in staff_boxes[page_first:]]):
                    tempo_found.append((page_first + mark.staff, mark))
                if progress is not None:
                    progress(done + 1, staffs)
            if not staffs:
                raise ReadError(422, "porte_bulunamadi",
                                "Görüntüde okunabilir porte bulunamadı. Sayfanın tamamı, düz ve net çekilmiş "
                                "olmalı.")
            out = os.path.join(scratch, "sonuc.musicxml")

            def write(tempo_marks: dict) -> tuple[str, dict[int, float], list[tuple[int, int]]]:
                confidence: dict[int, float] = {}
                staff_measures: list[tuple[int, int]] = []
                args = XmlGeneratorArguments(software=SOFTWARE, measure_confidence=confidence,
                                             staff_measures=staff_measures, tempo_marks=tempo_marks)
                generate_xml(args, [voice], "").write(out)
                with open(out, encoding="utf-8") as handle:
                    return handle.read(), confidence, staff_measures

            musicxml, confidence, staff_measures = write({})
            tempos = tempo_measures(tempo_found, staff_boxes, staff_measures)
            if tempos:
                # the measures are known only once written: write again with the tempo marks in their measures
                musicxml, confidence, staff_measures = write({t["olcu"]: (t["birim"], t["noktali"], t["deger"])
                                                              for t in tempos})
        written = {int(n) for n in re.findall(r'<measure number="(\d+)"', musicxml)}
        measures = [{"olcu": n, "guven": round(c, 3)} for n, c in sorted(confidence.items()) if n in written]
        weak = sum(1 for m in measures if m["guven"] < LOW_CONFIDENCE)
        if weak:
            warnings.append(f"{weak} ölçüde okuma güveni düşük; bu ölçüler kontrol edilmeli.")
        off = measures_off_meter(musicxml)
        if off:
            warnings.append(f"Ölçü {', '.join(map(str, off))}: süreler ölçü işaretini tutmuyor; kontrol edilmeli.")
        staff_list = []
        for number, ((page_no, box), (first, last)) in enumerate(zip(staff_boxes, staff_measures, strict=False), 1):
            measured = first <= last
            staff_list.append({"porte": number, "sayfa": page_no, "kutu": box,
                               "olcuBas": first if measured else None, "olcuSon": last if measured else None})
        return {"musicxml": musicxml, "modelVersion": MODEL_VERSION, "olcuGuven": measures, "uyarilar": warnings,
                "porteSayisi": staffs, "sayfaSayisi": len(pages), "toplamSayfa": total, "sayfalar": pages,
                "porteler": staff_list, "sayfaResimleri": page_info, "tempolar": tempos,
                "sureMs": round((time.perf_counter() - start) * 1000)}


app = FastAPI(title="Nota okuma servisi", docs_url=None, redoc_url=None, openapi_url=None)
reader: Reader | None = None
jobs: dict[str, dict] = {}
cancels: dict[str, threading.Event] = {}


def _reader() -> Reader:
    global reader  # noqa: PLW0603
    if reader is None:
        reader = Reader()
    return reader


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"hata": code, "mesaj": message})


@app.on_event("startup")
def _load() -> None:
    _reader()


@app.get("/v1/saglik")
def health() -> dict:
    return {"modelVersion": MODEL_VERSION, "cihaz": str(_reader().device)}


def _run(path: str, kind: str | None, pages: list[int], progress=None, cancel=None) -> dict:
    try:
        result = _reader().read(path, pages, progress, cancel)
        if kind:
            result["tur"] = kind
        return result
    finally:
        os.remove(path)


async def _upload(dosya: UploadFile) -> tuple[str, JSONResponse | None]:
    """Check and store an upload; returns its temporary path, or an error response."""
    suffix = TYPES.get((dosya.content_type or "").lower())
    if suffix is None:
        return "", _error(415, "desteklenmeyen_dosya", "Yalnız JPEG, PNG, WebP ya da PDF yüklenebilir.")
    data = await dosya.read()
    if len(data) > MAX_BYTES:
        return "", _error(413, "dosya_cok_buyuk", "Dosya 20 MB'tan büyük olamaz.")
    if not data:
        return "", _error(400, "bos_dosya", "Dosya boş.")
    handle, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(handle, "wb") as target:
        target.write(data)
    try:
        page_count(path)
    except Exception:  # noqa: BLE001
        os.remove(path)
        return "", _error(415, "desteklenmeyen_dosya", "PDF açılamadı.")
    return path, None


@app.post("/v1/sayfa-sayisi")
async def count_pages(dosya: UploadFile = File(...), x_omr_anahtar: str | None = Header(None)) -> JSONResponse:
    if KEY and x_omr_anahtar != KEY:
        return _error(401, "yetkisiz", "Geçersiz ya da eksik X-Omr-Anahtar.")
    path, error = await _upload(dosya)
    if error is not None:
        return error
    try:
        return JSONResponse(content={"sayfaSayisi": page_count(path)})
    finally:
        os.remove(path)


@app.post("/v1/omr")
async def read_score(
    dosya: UploadFile = File(...),
    tur: str | None = Form(None),
    sayfalar: str | None = Form(None),
    kuyruk: int = 0,
    x_omr_anahtar: str | None = Header(None),
) -> JSONResponse:
    if KEY and x_omr_anahtar != KEY:
        return _error(401, "yetkisiz", "Geçersiz ya da eksik X-Omr-Anahtar.")
    if tur not in (None, "", "basili", "el_yazisi"):
        return _error(400, "gecersiz_tur", "tur yalnız 'basili' ya da 'el_yazisi' olabilir.")
    path, error = await _upload(dosya)
    if error is not None:
        return error
    total = page_count(path)
    try:
        # a page range applies to PDFs only; an image is one page whatever is asked
        pages = parse_pages(sayfalar, total) if path.endswith(".pdf") else [1]
    except ReadError as error:
        os.remove(path)
        return _error(error.status, error.code, error.message)
    if kuyruk:
        job = uuid.uuid4().hex
        jobs[job] = {"durum": "isleniyor", "sayfaSayisi": len(pages), "toplamSayfa": total, "islenenSayfa": 0,
                     "islenenPorte": 0}
        cancels[job] = threading.Event()

        def progress(pages_done: int, staffs: int) -> None:
            jobs[job].update(islenenSayfa=pages_done, islenenPorte=staffs)

        def work() -> None:
            try:
                jobs[job] = {"durum": "tamam", **_run(path, tur, pages, progress, cancels[job])}
            except Cancelled:
                jobs[job] = {**jobs[job], "durum": "iptal"}
            except ReadError as error:
                jobs[job] = {"durum": "hata", "hata": error.code, "mesaj": error.message}
            except Exception:  # noqa: BLE001
                jobs[job] = {"durum": "hata", "hata": "ic_hata", "mesaj": "Okuma sırasında beklenmeyen bir hata oluştu."}
            finally:
                cancels.pop(job, None)

        threading.Thread(target=work, daemon=True).start()
        return JSONResponse(status_code=202, content={"isId": job, "sayfaSayisi": len(pages), "toplamSayfa": total})
    try:
        return JSONResponse(content=await _in_thread(path, tur, pages))
    except ReadError as error:
        return _error(error.status, error.code, error.message)
    except Exception:  # noqa: BLE001
        return _error(500, "ic_hata", "Okuma sırasında beklenmeyen bir hata oluştu.")


async def _in_thread(path: str, kind: str | None, pages: list[int]) -> dict:
    from starlette.concurrency import run_in_threadpool  # noqa: PLC0415

    return await run_in_threadpool(_run, path, kind, pages)


@app.get("/v1/omr/{job}")
def job_status(job: str, x_omr_anahtar: str | None = Header(None)) -> JSONResponse:
    if KEY and x_omr_anahtar != KEY:
        return _error(401, "yetkisiz", "Geçersiz ya da eksik X-Omr-Anahtar.")
    if job not in jobs:
        return _error(404, "is_bulunamadi", "Böyle bir iş yok.")
    return JSONResponse(content=jobs[job])


@app.delete("/v1/omr/{job}")
def cancel_job(job: str, x_omr_anahtar: str | None = Header(None)) -> JSONResponse:
    """A running job stops before its next staff and is kept as durum "iptal" (202); a finished, failed or cancelled
    job is removed (200)."""
    if KEY and x_omr_anahtar != KEY:
        return _error(401, "yetkisiz", "Geçersiz ya da eksik X-Omr-Anahtar.")
    if job not in jobs:
        return _error(404, "is_bulunamadi", "Böyle bir iş yok.")
    if job in cancels:
        cancels[job].set()
        return JSONResponse(status_code=202, content={"isId": job, "durum": "iptal ediliyor"})
    jobs.pop(job)
    return JSONResponse(content={"isId": job, "durum": "silindi"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("OMR_PORT", "8765")))
