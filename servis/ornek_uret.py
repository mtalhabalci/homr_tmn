"""Read the sample scores with the service's own reader and write what the app would receive:
servis/ornekler/<name>.musicxml and <name>.json (everything but the MusicXML). Also times each reading.

    python -m servis.ornek_uret
"""
import json
import os
import shutil
import tempfile

from servis.nota_servisi import Reader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "servis", "ornekler")
SAMPLES = {
    # Mus2, SymbTr test work (never in training): hicaz, key signature with flat and sharps, 9/4
    "basili_mus2_hicaz": "datasets/SymbTr-2.0.0/pdf/hicaz--sarki--agiraksak--firkatin_aldi--bimen_sen.pdf",
    # typeset (not Mus2), Erdinç Bal archive, page exam: no edition of the work in any training set
    "dizgi_erdincbal_acemasiran": "datasets/erdincbal-real/exam/pdf/hayir/"
                                  "EB01816__acemasiran--sarki--aksak--oldu_gonul--dede_efendi.pdf",
    # handwriting, Erdinç Bal archive, page exam (SymbTr val/test work)
    "el_yazisi_erdincbal_acemkurdi": "datasets/erdincbal-real/exam/pdf/evet/"
                                     "EB03676__acemkurdi--sarki--duyek--baharla_hazan--avni_anil.pdf",
}

os.makedirs(OUT, exist_ok=True)
reader = Reader()
print("cihaz:", reader.device)
for name, source in SAMPLES.items():
    # the service deletes nothing but its own temporary copy; read a copy as it would
    handle, path = tempfile.mkstemp(suffix=".pdf")
    os.close(handle)
    shutil.copyfile(os.path.join(ROOT, source), path)
    result = reader.read(path)
    os.remove(path)
    with open(os.path.join(OUT, name + ".musicxml"), "w", encoding="utf-8") as target:
        target.write(result["musicxml"])
    meta = {k: v for k, v in result.items() if k != "musicxml"} | {"kaynak": source}
    with open(os.path.join(OUT, name + ".json"), "w", encoding="utf-8") as target:
        json.dump(meta, target, ensure_ascii=False, indent=1)
    weak = [m for m in result["olcuGuven"] if m["guven"] < 0.5]
    print(f"{name}: {result['porteSayisi']} porte, {len(result['olcuGuven'])} ölçü, düşük güven {len(weak)}, "
          f"{result['sureMs']} ms, uyarı {result['uyarilar']}, homr geçiyor mu: {'homr' in result['musicxml'].lower()}")
