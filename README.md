# F1 Race Predictor — 2026 Season

FastF1 + scikit-learn ile Formula 1 yarış sonuç tahmini.

## Özellikler
- **Otomatik veri toplama** — FastF1 API'den 2023-2025 sezonlarının tüm verisi
- **Manuel giriş yok** — sürücü listesi, takım isimleri, puan tablosu otomatik
- **2026 grid** — 11 takım, 22 sürücü (Cadillac + Audi dahil)
- **GradientBoosting** — RandomForest yerine daha güçlü model
- **Herhangi bir yarış** — round belirtilerek geçmiş veya gelecek yarış tahmin edilebilir
- **JSON çıktı** — web arayüzü için hazır

## Kurulum
```bash
pip install fastf1 pandas numpy scikit-learn
```

## Kullanım
```bash
# Sıradaki yarışı tahmin et (otomatik bulur)
python f1_predictor.py

# Belirli bir yarış
python f1_predictor.py --round 3

# Geçmiş sezon tahmini
python f1_predictor.py --year 2025 --round 12
```

## Teknik Detaylar

### Veri Kaynakları
- **FastF1** — Kvalifikasyon süreleri, yarış sonuçları, hava durumu, pit stop
- **3 sezon eğitim verisi** — 2023, 2024, 2025 (60+ yarış, 1200+ satır)

### Feature'lar (23 adet)
| Feature | Açıklama |
|---------|----------|
| grid_position | Başlangıç pozisyonu |
| quali_time | En iyi kvalifikasyon süresi |
| delta_to_pole | Pole'a fark (saniye) |
| driver_form | Son 5 yarışın ort. bitiş pozisyonu |
| team_form | Takımın sezon ort. puanı |
| driver_dnf_rate | Sürücü bitirememe oranı |
| team_dnf_rate | Takım bitirememe oranı |
| races_so_far | Sürücünün o sezondaki yarış sayısı |
| ... | +15 diğer feature |

### Model
- **Kazanma**: GradientBoostingClassifier + CalibratedClassifierCV
- **Pozisyon**: GradientBoostingRegressor (1/position dönüşümü)
- **Final skor**: `win_prob × (1 + 3 × driver_pts_norm) / predicted_finish`

## 2026 Grid
| Takım | Sürücü 1 | Sürücü 2 |
|-------|----------|----------|
| McLaren | Lando Norris | Oscar Piastri |
| Mercedes | George Russell | Kimi Antonelli |
| Red Bull | Max Verstappen | Isack Hadjar |
| Ferrari | Charles Leclerc | Lewis Hamilton |
| Williams | Alexander Albon | Carlos Sainz |
| Aston Martin | Fernando Alonso | Lance Stroll |
| Alpine | Pierre Gasly | Franco Colapinto |
| Haas | Esteban Ocon | Oliver Bearman |
| Audi | Nico Hulkenberg | Gabriel Bortoleto |
| Racing Bulls | Liam Lawson | Arvid Lindblad |
| Cadillac | Sergio Perez | Valtteri Bottas |
