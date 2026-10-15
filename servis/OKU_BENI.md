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
| `OMR_TEMPO_MODEL` | `training/architecture/tempo/tempo_okuyucu.pt` | tempo okuyucunun ağırlığı (git dışında); yoksa tempo okunmaz |

## Uç noktalar

- `POST /v1/omr` — multipart `dosya` (JPEG, PNG, WebP, PDF; en çok 20 MB), isteğe bağlı `tur` (`basili` |
  `el_yazisi`; tek model ikisini de okur, yalnız geri yazılır) ve `sayfalar` (yalnız PDF: 1 tabanlı aralık listesi,
  ör. `"1-3,5"`; boşsa bütün sayfalar; resimde yok sayılır). Yalnız seçilen sayfalar okunur; MusicXML yalnız onları
  içerir ve ölçüler yine 1'den boşluksuz numaralanır. Yanıt: `musicxml`, `modelVersion`, `olcuGuven`
  (`[{olcu, guven}]`, ölçü numaraları MusicXML'dekiyle aynı), `uyarilar`, `porteSayisi`, `sayfaSayisi` (okunan),
  `toplamSayfa`, `sayfalar` (okunan sayfa numaraları), `sureMs`. `porteler`: okunan her porte için
  `{porte, sayfa, kutu: [x0, y0, x1, y1], olcuBas, olcuSon}`. Kutu, porte çizgilerinin o sayfanın resmindeki
  piksel konumudur: fotoğrafta yüklenen resmin kendisi, PDF'te servisin 1920 piksel genişlikte çizdiği sayfa.
  Ölçü numaraları MusicXML'dekiyle aynıdır; satır sonunda bölünen bir ölçü iki portede de görünür, ölçüsü
  olmayan bir portede ikisi de `null`'dır. `sayfaResimleri`: okunan her sayfa için `{sayfa, genislik, yukseklik,
  dpi}` (fotoğrafta `dpi` `null`). Kullanıcı düzeltmelerinin porte görüntüsüne eşlenmesi içindir.
  `tempolar`: sayfada yazılı tempo işaretleri (♪ = 112 gibi), her biri `{olcu, birim, noktali, deger, guven,
  porte, sayfa, kutu}` (kutu: işaretin sayfa resmindeki yeri); `birim` MusicXML adıyla (`eighth`, `quarter`, …). Aynı işaret MusicXML'de o ölçünün başında
  `<metronome>` ve dörtlük cinsinden `<sound tempo>` olarak yazılır. İşaret okunamadıysa liste boştur; uygulama
  usule göre varsayılan tempoyu kullanır. `?kuyruk=1` ile iş kuyruğa alınır ve hemen
  `{"isId", "sayfaSayisi", "toplamSayfa"}` döner (202). İşler yalnız bellekte durur; servis yeniden başlarsa iş
  kaybolur (`GET` 404 döner, uygulama yeniden gönderir).
- `GET /v1/omr/{isId}` — `{"durum": "isleniyor" | "tamam" | "hata" | "iptal", "islenenSayfa", "islenenPorte", ...}`;
  ilerleme her sayfa bitince güncellenir.
- `DELETE /v1/omr/{isId}` — süren iş bir sonraki porteden önce durur ve `durum: "iptal"` olarak kalır (202,
  `{"isId", "durum": "iptal ediliyor"}`); bitmiş, hatalı ya da iptal edilmiş iş silinir (200, `"silindi"`); yoksa 404.
- `POST /v1/sayfa-sayisi` — multipart `dosya` → `{"sayfaSayisi"}` (resim 1).
- `GET /v1/saglik` — `{"modelVersion", "cihaz"}`
- Hatalar: `{"hata", "mesaj"}`; 400 `gecersiz_tur` / `bos_dosya` / `gecersiz_sayfalar`, 401 `yetkisiz`, 413
  `dosya_cok_buyuk`, 415 `desteklenmeyen_dosya`, 422 `porte_bulunamadi`, 500 `ic_hata`.

## Ölçü güveni

Her satırın güveni, modelin seçtiği ritim, perde ve arıza olasılıklarının en küçüğüdür; ölçü işareti, ölçü
çizgisi, tekrar ve volta satırlarında yalnız işaretin kendisi sayılır. Ölçünün güveni içindeki satırların en
düşüğüdür. 0,5'in altı `uyarilar`'a yazılır. Örnek: basılı Mus2 ve dizgi sayfalarda düşük güvenli ölçü yok, el
yazısı bir sayfada 32 ölçünün 5'i. Süreleri ölçü işaretini tutmayan ölçüler de numaralarıyla `uyarilar`'a yazılır
(ilk ve son ölçü yalnız uzunsa; kısa olmaları ön vuruş olabilir).

## MusicXML yapısı

- Notasız ölçü yazılmaz; ölçüler 1'den boşluksuz numaralanır.
- İleri tekrar ölçünün `left` barline'ında, geri tekrar `right`'ta. Volta numaraları sırayla (1, 2); bitişi
  okunmamış birinci volta geri tekrarda, açık kalan son volta son ölçüde (`discontinue`) kapanır.
- `location="middle"` barline yazılmaz. Süsleme (çarpma) notaları `<grace/>` ile gelir.
- Arıza adları SMuFL özniteliğiyle: AEU (`accidentalKomaFlat` …) ve THM 2/3 koma (`accidental2CommaSharp`,
  `accidental3CommaSharp`, `accidental2CommaFlat`, `accidental3CommaFlat`).

## Süre (14.10.2026, bu bilgisayar)

| Sayfa | Porte | Ekran kartı | İşlemci |
|---|---:|---:|---:|
| Mus2, 2 sayfa (hicaz) | 16 | 13,6 sn | – |
| Dizgi, 1 sayfa | 8 | 6,6–7,0 sn | 11,2 sn |
| El yazısı, 1 sayfa | 8 | 7,0 sn | – |

## Örnekler

`servis/ornekler/`: aynı servisin okuduğu üç sayfa (Mus2 basılı hicaz, Erdinç Bal dizgi, Erdinç Bal el yazısı;
hiçbiri eğitimde yok), MusicXML ve yanıtın geri kalanı (`.json`). `python -m servis.ornek_uret` yeniden üretir.
