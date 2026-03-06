"""
F1 Race Predictor — 2026 Season
================================
FastF1 + scikit-learn ile Formula 1 yarış sonuç tahmini.
Tüm veriler API'den otomatik çekilir, manuel giriş yok.

Kullanım:
  python f1_predictor.py                      # Sıradaki yarışı tahmin et
  python f1_predictor.py --round 3            # Belirli round'u tahmin et
  python f1_predictor.py --year 2025 --round 12  # Geçmiş yarış tahmini

Gereksinimler:
  pip install fastf1 pandas numpy scikit-learn
"""

import fastf1
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OrdinalEncoder
import warnings
import argparse
import json
import os
from datetime import datetime

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
# CACHE & CONFIG
# ═══════════════════════════════════════════════════════════════
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "f1_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

# Eğitim için kullanılacak geçmiş sezonlar
TRAINING_YEARS = [2023, 2024, 2025]

# ═══════════════════════════════════════════════════════════════
# 2026 GRID — API'den alınamayan bilgiler (sezon başı sabit)
# Bu veriler formula1.com ve Wikipedia'dan doğrulanmıştır
# ═══════════════════════════════════════════════════════════════
GRID_2026 = {
    # team: [driver1, driver2]
    "McLaren":       ["Lando Norris", "Oscar Piastri"],
    "Mercedes":      ["George Russell", "Andrea Kimi Antonelli"],
    "Red Bull Racing":["Max Verstappen", "Isack Hadjar"],
    "Ferrari":       ["Charles Leclerc", "Lewis Hamilton"],
    "Williams":      ["Alexander Albon", "Carlos Sainz"],
    "Aston Martin":  ["Fernando Alonso", "Lance Stroll"],
    "Alpine":        ["Pierre Gasly", "Franco Colapinto"],
    "Haas F1 Team":  ["Esteban Ocon", "Oliver Bearman"],
    "Audi":          ["Nico Hulkenberg", "Gabriel Bortoleto"],
    "Racing Bulls":  ["Liam Lawson", "Arvid Lindblad"],
    "Cadillac":      ["Sergio Perez", "Valtteri Bottas"],
}

# Takım isim değişiklik haritası (FastF1'deki isim ↔ gerçek isim)
TEAM_ALIASES = {
    "Kick Sauber": "Audi",
    "Sauber": "Audi",
    "AlphaTauri": "Racing Bulls",
    "RB": "Racing Bulls",
    "Alfa Romeo": "Audi",
}

def normalize_team(name):
    """FastF1'den gelen takım ismini 2026 ismine çevir."""
    return TEAM_ALIASES.get(name, name)


class F1Predictor:
    """
    Formula 1 yarış sonuç tahmincisi.
    
    - Geçmiş 3 sezonun tüm yarış verilerini FastF1'den çeker
    - GradientBoosting ile kazanma olasılığı + bitiş pozisyonu tahmin eder
    - Herhangi bir yarış round'u için tahmin üretir
    """
    
    def __init__(self):
        self.clf = None  # Kazanma sınıflandırıcısı
        self.reg = None  # Pozisyon regresörü
        self.data = pd.DataFrame()
        self.driver_points = {}   # Sezon toplam puanları
        self.team_points = {}
        self.is_trained = False
        self.feature_cols = []
        
        # Ordinal encoder (LabelEncoder yerine — bilinmeyen değer desteği)
        self.driver_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self.team_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self.circuit_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    
    # ─────────────────────────────────────────────
    # YARDIMCI METODLAR
    # ─────────────────────────────────────────────
    def _safe_load_session(self, year, rnd, session_type):
        """Session yükle, hata varsa None dön."""
        try:
            s = fastf1.get_session(year, rnd, session_type)
            s.load()
            return s
        except Exception as e:
            print(f"  ⚠ {year} R{rnd} {session_type} yüklenemedi: {e}")
            return None
    
    def _get_weather(self, session):
        """Session'dan ortalama hava durumu verisi çek."""
        try:
            w = session.weather_data
            if w is None or w.empty:
                return np.nan, np.nan
            return float(w["AirTemp"].mean()), float(w["TrackTemp"].mean())
        except Exception:
            return np.nan, np.nan
    
    def _best_quali_time(self, quali_row):
        """Sürücünün en iyi kvalifikasyon süresini bul (Q3 > Q2 > Q1)."""
        for q in ["Q3", "Q2", "Q1"]:
            val = quali_row.get(q)
            if pd.notna(val):
                return val.total_seconds() if hasattr(val, "total_seconds") else 0.0
        return 0.0
    
    def _latest_completed_round(self, year):
        """Bir sezondaki en son tamamlanmış yarışın round numarası."""
        try:
            schedule = fastf1.get_event_schedule(year)
            now = pd.Timestamp.now(tz="UTC")
            latest = 0
            for _, ev in schedule.iterrows():
                race_date = ev.get("Session5Date") or ev.get("Session3Date")
                if race_date and now > race_date:
                    latest = max(latest, ev["RoundNumber"])
            return latest
        except Exception:
            return 0
    
    # ─────────────────────────────────────────────
    # VERİ TOPLAMA
    # ─────────────────────────────────────────────
    def collect_data(self, target_year=2026, target_round=None):
        """
        Eğitim verisi topla.
        - TRAINING_YEARS'daki tüm sezonların tamamlanmış yarışlarını çeker
        - target_year'ın target_round öncesindeki yarışlarını çeker
        """
        records = []
        
        for year in TRAINING_YEARS:
            max_rnd = self._latest_completed_round(year)
            if max_rnd == 0:
                print(f"  {year}: Tamamlanmış yarış yok, atlanıyor.")
                continue
            
            print(f"\n📥 {year} sezonu yükleniyor ({max_rnd} yarış)...")
            for rnd in range(1, max_rnd + 1):
                rows = self._extract_round_data(year, rnd)
                if rows:
                    records.extend(rows)
                    print(f"  ✓ R{rnd:02d} — {len(rows)} sürücü")
        
        # Hedef yılın verilerini de çek (target_round öncesi)
        if target_year not in TRAINING_YEARS:
            max_rnd = self._latest_completed_round(target_year)
            cutoff = (target_round - 1) if target_round else max_rnd
            usable = min(max_rnd, cutoff)
            
            if usable > 0:
                print(f"\n📥 {target_year} sezonu yükleniyor ({usable} yarış)...")
                for rnd in range(1, usable + 1):
                    rows = self._extract_round_data(target_year, rnd)
                    if rows:
                        records.extend(rows)
                        print(f"  ✓ R{rnd:02d} — {len(rows)} sürücü")
        
        self.data = pd.DataFrame(records)
        print(f"\n📊 Toplam eğitim verisi: {len(self.data)} satır")
        
        # Sezon puanlarını hesapla
        if not self.data.empty:
            latest_year = self.data["year"].max()
            season_data = self.data[self.data["year"] == latest_year]
            self.driver_points = season_data.groupby("driver")["points"].sum().to_dict()
            self.team_points = season_data.groupby("team")["points"].sum().to_dict()
        
        return self.data
    
    def _extract_round_data(self, year, rnd):
        """Bir yarışın tüm sürücü verilerini çıkar."""
        race = self._safe_load_session(year, rnd, "R")
        quali = self._safe_load_session(year, rnd, "Q")
        if race is None or race.results.empty:
            return []
        
        race_air, race_track = self._get_weather(race)
        quali_air, quali_track = self._get_weather(quali) if quali else (np.nan, np.nan)
        
        circuit = race.event.get("Location", "Unknown")
        qres = quali.results if quali and not quali.results.empty else pd.DataFrame()
        
        rows = []
        for _, r in race.results.iterrows():
            # Kvalifikasyon verisi
            best_q, q_pos = 0.0, 20
            if not qres.empty:
                dq = qres[qres["Abbreviation"] == r["Abbreviation"]]
                if not dq.empty:
                    best_q = self._best_quali_time(dq.iloc[0])
                    q_pos = dq.iloc[0].get("Position", 20)
            
            team = normalize_team(r.get("TeamName", "Unknown"))
            status = r.get("Status", "")
            finished = 1 if "Finished" in str(status) or status.startswith("+") else 0
            
            rows.append({
                "year": year,
                "round": rnd,
                "circuit": circuit,
                "driver": r.get("Abbreviation", "???"),
                "driver_name": r.get("FullName", "Unknown"),
                "team": team,
                "grid_position": r.get("GridPosition", 20),
                "finish_position": r.get("Position", 20),
                "points": r.get("Points", 0),
                "quali_position": q_pos,
                "quali_time": best_q,
                "is_winner": 1 if r.get("Position") == 1 else 0,
                "status": status,
                "finished": finished,
                "race_air_temp": race_air,
                "race_track_temp": race_track,
                "quali_air_temp": quali_air,
                "quali_track_temp": quali_track,
            })
        
        return rows
    
    # ─────────────────────────────────────────────
    # FEATURE ENGINEERING
    # ─────────────────────────────────────────────
    def _engineer_features(self, df):
        """Tüm feature'ları hesapla."""
        df = df.sort_values(["driver", "year", "round"]).reset_index(drop=True)
        
        # Sürücü deneyimi
        df["races_so_far"] = df.groupby("driver").cumcount()
        
        # Takım değişikliği
        df["prev_team"] = df.groupby("driver")["team"].shift(1)
        df["team_changed"] = (df["team"] != df["prev_team"]).astype(int).fillna(0).astype(int)
        
        # Rookie
        first_year = df.groupby("driver")["year"].transform("min")
        df["rookie"] = (df["year"] == first_year).astype(int)
        
        # Kvalifikasyon vs grid farkı (penalty tespiti)
        df["quali_vs_grid"] = df["quali_position"] - df["grid_position"]
        
        # Takım ortalama puanı (expanding mean, önceki yarışlardan)
        df = df.sort_values(["year", "round"])
        df["team_form"] = (
            df.groupby(["year", "team"])["points"]
              .transform(lambda s: s.shift(1).expanding().mean())
        ).fillna(0)
        
        # Sürücü form (son 5 yarışın ortalama bitiş pozisyonu)
        df["driver_form"] = (
            df.groupby("driver")["finish_position"]
              .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
        ).fillna(10)
        
        # DNF oranları
        df["driver_dnf_rate"] = (
            df.groupby("driver")["finished"]
              .transform(lambda s: 1 - s.shift(1).expanding().mean())
        ).fillna(0)
        df["team_dnf_rate"] = (
            df.groupby("team")["finished"]
              .transform(lambda s: 1 - s.shift(1).expanding().mean())
        ).fillna(0)
        
        # Pole'a uzaklık
        df["delta_to_pole"] = 0.0
        for (y, rnd), grp in df.groupby(["year", "round"]):
            valid = grp["quali_time"]
            pole = valid[valid > 0].min()
            if pd.notna(pole) and pole > 0:
                df.loc[grp.index, "delta_to_pole"] = grp["quali_time"] - pole
        
        # Sezon puanları (normalize)
        max_dp = max(self.driver_points.values()) if self.driver_points else 1
        max_tp = max(self.team_points.values()) if self.team_points else 1
        df["driver_season_pts"] = df["driver"].map(self.driver_points).fillna(0)
        df["team_season_pts"] = df["team"].map(self.team_points).fillna(0)
        df["driver_pts_norm"] = df["driver_season_pts"] / (max_dp or 1)
        df["team_pts_norm"] = df["team_season_pts"] / (max_tp or 1)
        
        # Hava durumu — median ile doldur
        for col in ["race_air_temp", "race_track_temp", "quali_air_temp", "quali_track_temp"]:
            median = df[col].median()
            df[col] = df[col].fillna(median if pd.notna(median) else 25)
        
        return df
    
    def _prepare_features(self, df):
        """Feature matrix ve sütun isimlerini döndür."""
        df = self._engineer_features(df)
        
        # Encoding
        df["driver_enc"] = self.driver_enc.fit_transform(df[["driver"]]).ravel()
        df["team_enc"] = self.team_enc.fit_transform(df[["team"]]).ravel()
        df["circuit_enc"] = self.circuit_enc.fit_transform(df[["circuit"]]).ravel()
        
        self.feature_cols = [
            "driver_enc", "team_enc", "circuit_enc",
            "grid_position", "quali_position", "quali_time", "delta_to_pole",
            "quali_vs_grid", "team_changed", "team_form", "driver_form",
            "rookie", "races_so_far",
            "driver_season_pts", "team_season_pts", "driver_pts_norm", "team_pts_norm",
            "driver_dnf_rate", "team_dnf_rate",
            "race_air_temp", "race_track_temp", "quali_air_temp", "quali_track_temp",
        ]
        
        return df, self.feature_cols
    
    # ─────────────────────────────────────────────
    # MODEL EĞİTİMİ
    # ─────────────────────────────────────────────
    def train(self):
        """Modelleri eğit."""
        if self.data.empty:
            raise ValueError("Önce collect_data() çağır.")
        
        df, feats = self._prepare_features(self.data.copy())
        X = df[feats]
        
        # Kazanma sınıflandırıcısı
        y_win = df["is_winner"]
        
        # Bitiş pozisyonu regresörü (1/pos dönüşümü)
        finish = df["finish_position"].astype(float).clip(lower=1)
        y_reg = 1.0 / finish
        
        # Sample weights — son sezon ağırlıklı
        latest_year = df["year"].max()
        year_weight = np.where(df["year"] == latest_year, 3.0, 1.0)
        strength = 1 + df["driver_pts_norm"] + 0.5 * df["team_pts_norm"]
        sample_weights = year_weight * strength.values
        
        print("\n🏋️ Model eğitiliyor...")
        
        # Classifier
        base_clf = GradientBoostingClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42
        )
        cv_scores = cross_val_score(base_clf, X, y_win, cv=5, scoring="accuracy",
                                     fit_params={"sample_weight": sample_weights})
        print(f"  Kazanma CV Accuracy: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
        
        base_clf.fit(X, y_win, sample_weight=sample_weights)
        self.clf = CalibratedClassifierCV(base_clf, cv=5, method="isotonic")
        self.clf.fit(X, y_win, sample_weight=sample_weights)
        
        # Regressor
        self.reg = GradientBoostingRegressor(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42
        )
        self.reg.fit(X, y_reg, sample_weight=sample_weights)
        
        # MAE
        y_hat = self.reg.predict(X)
        y_hat_pos = 1.0 / np.maximum(y_hat, 1e-6)
        y_true_pos = 1.0 / np.maximum(y_reg, 1e-6)
        mae = mean_absolute_error(y_true_pos, y_hat_pos)
        print(f"  Pozisyon MAE (train): {mae:.2f}")
        
        # Top feature importances
        importances = self.reg.feature_importances_
        top_idx = np.argsort(importances)[::-1][:8]
        print("  Top features:", ", ".join(f"{feats[i]}({importances[i]:.3f})" for i in top_idx))
        
        self.is_trained = True
        print("  ✅ Eğitim tamamlandı.")
    
    # ─────────────────────────────────────────────
    # TAHMİN
    # ─────────────────────────────────────────────
    def predict(self, year=2026, rnd=1):
        """Belirli bir yarış için tahmin üret."""
        if not self.is_trained:
            raise ValueError("Önce train() çağır.")
        
        print(f"\n🏁 {year} Round {rnd} tahmini yapılıyor...")
        
        # Kvalifikasyon verisini çek
        quali = self._safe_load_session(year, rnd, "Q")
        if quali is None or quali.results.empty:
            print("  ⚠ Kvalifikasyon verisi yok — grid pozisyonlarından tahmin yapılacak.")
            return self._predict_from_grid(year, rnd)
        
        qres = quali.results
        circuit = quali.event.get("Location", "Unknown")
        q_air, q_track = self._get_weather(quali)
        
        # Pole zamanı
        pole_time = None
        times = []
        for _, row in qres.iterrows():
            t = self._best_quali_time(row)
            if t > 0:
                times.append(t)
        pole_time = min(times) if times else None
        
        # Her sürücü için feature vektörü oluştur
        predictions = []
        max_dp = max(self.driver_points.values()) if self.driver_points else 1
        max_tp = max(self.team_points.values()) if self.team_points else 1
        
        for _, row in qres.iterrows():
            driver_code = row.get("Abbreviation", "???")
            driver_name = row.get("FullName", "Unknown")
            team = normalize_team(row.get("TeamName", "Unknown"))
            
            best_q = self._best_quali_time(row)
            q_pos = row.get("Position", 20)
            grid_pos = q_pos  # Varsayılan: quali = grid
            
            delta = (best_q - pole_time) if (pole_time and best_q > 0) else 0.0
            
            # Geçmiş verilerden sürücü istatistikleri
            drv_hist = self.data[(self.data["driver"] == driver_code)]
            drv_this_year = drv_hist[drv_hist["year"] == year]
            team_this_year = self.data[(self.data["team"] == team) & (self.data["year"] == year)]
            
            races_so_far = len(drv_this_year)
            driver_form = drv_hist["finish_position"].tail(5).mean() if not drv_hist.empty else 10.0
            team_form_val = team_this_year["points"].mean() if not team_this_year.empty else 0.0
            
            drv_dnf = 1 - drv_hist["finished"].mean() if not drv_hist.empty else 0.0
            team_dnf = 1 - self.data[self.data["team"] == team]["finished"].mean() if not self.data.empty else 0.0
            
            d_pts = self.driver_points.get(driver_code, 0)
            t_pts = self.team_points.get(team, 0)
            
            # Encoding — bilinmeyen değerler -1 olur
            try:
                d_enc = self.driver_enc.transform([[driver_code]])[0][0]
            except:
                d_enc = -1
            try:
                t_enc = self.team_enc.transform([[team]])[0][0]
            except:
                t_enc = -1
            try:
                c_enc = self.circuit_enc.transform([[circuit]])[0][0]
            except:
                c_enc = -1
            
            feat_vec = [
                d_enc, t_enc, c_enc,
                grid_pos, q_pos, best_q, delta,
                q_pos - grid_pos,  # quali_vs_grid
                0,  # team_changed (sezon içi nadiren)
                team_form_val, driver_form,
                1 if races_so_far == 0 else 0,  # rookie
                races_so_far,
                d_pts, t_pts,
                d_pts / (max_dp or 1), t_pts / (max_tp or 1),
                drv_dnf, team_dnf,
                q_air if pd.notna(q_air) else 25,
                q_track if pd.notna(q_track) else 35,
                q_air if pd.notna(q_air) else 25,
                q_track if pd.notna(q_track) else 35,
            ]
            
            win_prob = self.clf.predict_proba([feat_vec])[0][1]
            reg_pred = self.reg.predict([feat_vec])[0]
            pred_finish = 1.0 / max(reg_pred, 1e-6)
            
            predictions.append({
                "driver": driver_name,
                "team": team,
                "grid": int(grid_pos),
                "predicted_finish": round(pred_finish, 1),
                "win_probability": round(win_prob * 100, 2),
                "score": win_prob * (1 + 3 * d_pts / (max_dp or 1)) / max(pred_finish, 1),
            })
        
        # Sırala ve pozisyon ata
        result = pd.DataFrame(predictions)
        result = result.sort_values("score", ascending=False).reset_index(drop=True)
        result["predicted_position"] = result.index + 1
        
        # Çıktı
        print(f"\n{'═' * 70}")
        print(f"  🏆 {year} Round {rnd} — {circuit} — Tahmini Yarış Sonucu")
        print(f"{'═' * 70}")
        display = result[["predicted_position", "driver", "team", "grid", "predicted_finish", "win_probability"]]
        display.columns = ["Pos", "Sürücü", "Takım", "Grid", "Tahmini Bitiş", "Kazanma %"]
        print(display.to_string(index=False))
        
        if pole_time:
            m, s = divmod(pole_time, 60)
            print(f"\n  ⏱ Pole zamanı: {int(m)}:{s:06.3f}")
        
        # JSON kaydet
        output_file = f"prediction_{year}_R{rnd:02d}.json"
        result.to_json(output_file, orient="records", indent=2, force_ascii=False)
        print(f"\n  💾 Kaydedildi: {output_file}")
        
        return result
    
    def _predict_from_grid(self, year, rnd):
        """Kvalifikasyon verisi yoksa 2026 grid'inden tahmin yap."""
        print("  Grid bilgilerinden basit tahmin yapılıyor...")
        
        predictions = []
        pos = 1
        for team, drivers in GRID_2026.items():
            for driver_name in drivers:
                d_pts = self.driver_points.get(driver_name, 0)
                max_dp = max(self.driver_points.values()) if self.driver_points else 1
                
                predictions.append({
                    "driver": driver_name,
                    "team": team,
                    "grid": pos,
                    "predicted_finish": pos,
                    "win_probability": max(0, round((1 - pos / 22) * 20, 2)),
                    "score": 1 / pos,
                    "predicted_position": pos,
                })
                pos += 1
        
        result = pd.DataFrame(predictions)
        print("\n  ⚠ Bu tahmin yalnızca grid sırasına dayanır — gerçek bir ML tahmini değil.")
        print(result[["predicted_position", "driver", "team"]].to_string(index=False))
        return result


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="F1 Yarış Sonuç Tahmincisi")
    parser.add_argument("--year", type=int, default=2026, help="Hedef yıl")
    parser.add_argument("--round", type=int, default=None, help="Hedef round (None = sıradaki)")
    args = parser.parse_args()
    
    target_year = args.year
    target_round = args.round
    
    # Sıradaki yarışı otomatik bul
    if target_round is None:
        try:
            schedule = fastf1.get_event_schedule(target_year)
            now = pd.Timestamp.now(tz="UTC")
            for _, ev in schedule.iterrows():
                race_date = ev.get("Session5Date") or ev.get("Session3Date")
                if race_date and now < race_date:
                    target_round = ev["RoundNumber"]
                    print(f"🎯 Sıradaki yarış: R{target_round} — {ev.get('EventName', '?')}")
                    break
            if target_round is None:
                target_round = 1
                print("  Sezon takvimi alınamadı, R1 kullanılıyor.")
        except Exception as e:
            target_round = 1
            print(f"  Takvim hatası: {e}, R1 kullanılıyor.")
    
    print(f"\n{'━' * 50}")
    print(f"  F1 PREDICTOR — {target_year} Round {target_round}")
    print(f"{'━' * 50}")
    
    predictor = F1Predictor()
    predictor.collect_data(target_year=target_year, target_round=target_round)
    
    if predictor.data.empty:
        print("\n❌ Yeterli veri yok. FastF1 cache'i ve internet bağlantısını kontrol edin.")
        return
    
    predictor.train()
    predictor.predict(year=target_year, rnd=target_round)


if __name__ == "__main__":
    main()
