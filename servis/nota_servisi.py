"""Photo or PDF of a makam score in, MusicXML out: the HTTP service the Musiki Akademisi app calls.

    python -m servis.nota_servisi                 listens on 127.0.0.1:OMR_PORT (default 8765)

    POST /v1/omr            multipart: dosya (JPEG, PNG, WebP or PDF, at most 20 MB), tur (basili | el_yazisi,
                            optional; one model reads both, so it is only echoed back)
                            ?kuyruk=1 queues the job and answers {"isId"} at once
    GET  /v1/omr/{isId}     {"durum": "isleniyor" | "tamam" | "hata", ...}
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
from concurrent.futures import Future

import cv2
import fitz
import torch
from fastapi import FastAPI, File, Form, Header, UploadFile
from fastapi.responses import JSONResponse

from homr import main as reader_main
from homr.main import ProcessingConfig, detect_staffs_in_image
from homr.music_xml_generator import XmlGeneratorArguments, generate_xml
from homr.staff_parsing import prepare_staff_image
from homr.staff_regions import StaffRegions
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


class ReadError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


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

    def _pages(self, path: str, scratch: str) -> list[str]:
        if not path.endswith(".pdf"):
            return [path]
        images = []
        with fitz.open(path) as document:
            for page in document:
                zoom = PAGE_WIDTH / page.rect.width
                image = os.path.join(scratch, f"sayfa-{page.number + 1}.png")
                page.get_pixmap(matrix=fitz.Matrix(zoom, zoom)).save(image)
                images.append(image)
        return images

    def read(self, path: str) -> dict:
        start = time.perf_counter()
        warnings: list[str] = []
        voice: list[EncodedSymbol] = []
        staffs = 0
        with self.lock, tempfile.TemporaryDirectory() as scratch:
            pages = self._pages(path, scratch)
            for page_no, image in enumerate(pages, start=1):
                where = f"{page_no}. sayfa" if len(pages) > 1 else "Sayfa"
                try:
                    multi_staffs, preprocessed, debug, _ = detect_staffs_in_image(image, self.processing)
                except Exception:  # noqa: BLE001
                    warnings.append(f"{where}: porte bulunamadı.")
                    continue
                regions = StaffRegions(multi_staffs)
                found = sorted((s for multi in multi_staffs for s in multi.staffs), key=lambda s: s.min_y)
                for number, staff in enumerate(found):
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
                    staffs += 1
            if not staffs:
                raise ReadError(422, "porte_bulunamadi",
                                "Görüntüde okunabilir porte bulunamadı. Sayfanın tamamı, düz ve net çekilmiş "
                                "olmalı.")
            confidence: dict[int, float] = {}
            args = XmlGeneratorArguments(software=SOFTWARE, measure_confidence=confidence)
            out = os.path.join(scratch, "sonuc.musicxml")
            generate_xml(args, [voice], "").write(out)
            with open(out, encoding="utf-8") as handle:
                musicxml = handle.read()
        written = {int(n) for n in re.findall(r'<measure number="(\d+)"', musicxml)}
        measures = [{"olcu": n, "guven": round(c, 3)} for n, c in sorted(confidence.items()) if n in written]
        weak = sum(1 for m in measures if m["guven"] < LOW_CONFIDENCE)
        if weak:
            warnings.append(f"{weak} ölçüde okuma güveni düşük; bu ölçüler kontrol edilmeli.")
        return {"musicxml": musicxml, "modelVersion": MODEL_VERSION, "olcuGuven": measures, "uyarilar": warnings,
                "porteSayisi": staffs, "sureMs": round((time.perf_counter() - start) * 1000)}


app = FastAPI(title="Nota okuma servisi", docs_url=None, redoc_url=None, openapi_url=None)
reader: Reader | None = None
jobs: dict[str, dict] = {}


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


def _run(path: str, kind: str | None) -> dict:
    try:
        result = _reader().read(path)
        if kind:
            result["tur"] = kind
        return result
    finally:
        os.remove(path)


@app.post("/v1/omr")
async def read_score(
    dosya: UploadFile = File(...),
    tur: str | None = Form(None),
    kuyruk: int = 0,
    x_omr_anahtar: str | None = Header(None),
) -> JSONResponse:
    if KEY and x_omr_anahtar != KEY:
        return _error(401, "yetkisiz", "Geçersiz ya da eksik X-Omr-Anahtar.")
    if tur not in (None, "", "basili", "el_yazisi"):
        return _error(400, "gecersiz_tur", "tur yalnız 'basili' ya da 'el_yazisi' olabilir.")
    suffix = TYPES.get((dosya.content_type or "").lower())
    if suffix is None:
        return _error(415, "desteklenmeyen_dosya", "Yalnız JPEG, PNG, WebP ya da PDF yüklenebilir.")
    data = await dosya.read()
    if len(data) > MAX_BYTES:
        return _error(413, "dosya_cok_buyuk", "Dosya 20 MB'tan büyük olamaz.")
    if not data:
        return _error(400, "bos_dosya", "Dosya boş.")
    handle, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(handle, "wb") as target:
        target.write(data)
    if kuyruk:
        job = uuid.uuid4().hex
        jobs[job] = {"durum": "isleniyor"}

        def work() -> None:
            try:
                jobs[job] = {"durum": "tamam", **_run(path, tur)}
            except ReadError as error:
                jobs[job] = {"durum": "hata", "hata": error.code, "mesaj": error.message}
            except Exception:  # noqa: BLE001
                jobs[job] = {"durum": "hata", "hata": "ic_hata", "mesaj": "Okuma sırasında beklenmeyen bir hata oluştu."}

        threading.Thread(target=work, daemon=True).start()
        return JSONResponse(status_code=202, content={"isId": job})
    try:
        return JSONResponse(content=await _in_thread(path, tur))
    except ReadError as error:
        return _error(error.status, error.code, error.message)
    except Exception:  # noqa: BLE001
        return _error(500, "ic_hata", "Okuma sırasında beklenmeyen bir hata oluştu.")


async def _in_thread(path: str, kind: str | None) -> dict:
    from starlette.concurrency import run_in_threadpool  # noqa: PLC0415

    return await run_in_threadpool(_run, path, kind)


@app.get("/v1/omr/{job}")
def job_status(job: str, x_omr_anahtar: str | None = Header(None)) -> JSONResponse:
    if KEY and x_omr_anahtar != KEY:
        return _error(401, "yetkisiz", "Geçersiz ya da eksik X-Omr-Anahtar.")
    if job not in jobs:
        return _error(404, "is_bulunamadi", "Böyle bir iş yok.")
    return JSONResponse(content=jobs[job])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("OMR_PORT", "8765")))
