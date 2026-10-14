# Nota okuma servisi

Musiki Akademisi uygulamasının çağırdığı servis: fotoğraf ya da PDF alır, MusicXML döndürür. Model tezin son
modelidir (v20 ağırlıkları + çözümleme kuralları). Uygulamada ve servis yanıtlarında kaynak aldığımız açık kaynak
projenin adı geçmez (Talha'nın kuralı); MusicXML'deki `<software>` alanı `OMR_YAZILIM` ile verilir.

## Çalıştırma

    cd C:\projects\yl\homr
    C:\projects\yl\venv\Scripts\python.exe -m servis.nota_servisi

`127.0.0.1:8765` adresinde dinler. Model ilk açılışta yüklenir (~10 sn).

| Ortam değişkeni | Varsayılan | Ne işe yarar |
|---|---|---|
| `OMR_MODEL` | `training/architecture/transformer/pytorch_model_1-111a84bed…pth` | ağırlık dosyası (git dışında) |
| `OMR_MODEL_SURUM` | `v20+kurallar-2026-10-10` | yanıttaki `modelVersion` |
| `OMR_YAZILIM` | `Musiki Akademisi OMR` | MusicXML `<software>` |
| `OMR_ANAHTAR` | boş | doluysa `X-Omr-Anahtar` başlığı buna eşit olmalı |
| `OMR_PORT` | `8765` | port |
| `OMR_GPU_SEGMENTASYON` | kapalı | `1`: porte bulma da ekran kartında |

## Uç noktalar

- `POST /v1/omr` — multipart `dosya` (JPEG, PNG, WebP, PDF; en çok 20 MB), isteğe bağlı `tur` (`basili` |
  `el_yazisi`; tek model ikisini de okur, yalnız geri yazılır). Yanıt: `musicxml`, `modelVersion`, `olcuGuven`
  (`[{olcu, guven}]`, ölçü numaraları MusicXML'dekiyle aynı), `uyarilar`, `porteSayisi`, `sureMs`.
  `?kuyruk=1` ile iş kuyruğa alınır ve hemen `{"isId"}` döner (202).
- `GET /v1/omr/{isId}` — `{"durum": "isleniyor" | "tamam" | "hata", ...}`
- `GET /v1/saglik` — `{"modelVersion", "cihaz"}`
- Hatalar: `{"hata", "mesaj"}`; 400 `gecersiz_tur` / `bos_dosya`, 401 `yetkisiz`, 413 `dosya_cok_buyuk`,
  415 `desteklenmeyen_dosya`, 422 `porte_bulunamadi`, 500 `ic_hata`.

## Ölçü güveni

Her satırın güveni, modelin seçtiği ritim, perde ve arıza olasılıklarının en küçüğüdür; ölçü işareti, ölçü
çizgisi, tekrar ve volta satırlarında yalnız işaretin kendisi sayılır. Ölçünün güveni içindeki satırların en
düşüğüdür. 0,5'in altı `uyarilar`'a yazılır. Örnek: basılı Mus2 ve dizgi sayfalarda düşük güvenli ölçü yok, el
yazısı bir sayfada 34 ölçünün 5'i.

## Süre (14.10.2026, bu bilgisayar)

| Sayfa | Porte | Ekran kartı | İşlemci |
|---|---:|---:|---:|
| Mus2, 2 sayfa (hicaz) | 16 | 13,6 sn | – |
| Dizgi, 1 sayfa | 8 | 6,6–7,0 sn | 11,2 sn |
| El yazısı, 1 sayfa | 8 | 7,0 sn | – |

## Örnekler

`servis/ornekler/`: aynı servisin okuduğu üç sayfa (Mus2 basılı hicaz, Erdinç Bal dizgi, Erdinç Bal el yazısı;
hiçbiri eğitimde yok), MusicXML ve yanıtın geri kalanı (`.json`). `python -m servis.ornek_uret` yeniden üretir.
