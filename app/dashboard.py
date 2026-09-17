"""Read-only research dashboard. Never downloads data or trains a model."""

from __future__ import annotations

import json
import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

SERIES = "f1"
TITLE = "F1 · Yarış araştırması"
ROOT = Path(__file__).resolve().parents[1]
MAX_REPORT_BYTES = 25 * 1024 * 1024


def read_report(raw: bytes) -> dict:
    """Validate the public report envelope before touching chart data."""
    if len(raw) > MAX_REPORT_BYTES:
        raise ValueError("Rapor 25 MB sınırını aşıyor. Daha küçük bir çalışma seçin.")
    report = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(report, dict):
        raise ValueError("Rapor bir JSON nesnesi olmalı.")
    if report.get("schema_version") != 1:
        raise ValueError("Desteklenen rapor sürümü: schema_version = 1.")
    if report.get("series") != SERIES:
        raise ValueError(f"Bu uygulama yalnız {SERIES.upper()} raporlarını açar.")
    if report.get("data_kind") not in {"historical", "synthetic"}:
        raise ValueError("data_kind, historical veya synthetic olmalı.")
    for field in ("summary", "metrics", "predictions"):
        rows = report.get(field, [])
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError(f"{field} alanı JSON nesnelerinden oluşan bir liste olmalı.")
    for row in report.get("predictions", []):
        if any(not isinstance(row.get(key), (str, int)) for key in ("event_id", "driver", "model")):
            raise ValueError("Her tahmin event_id, driver ve model alanlarını içermeli.")
    if not isinstance(report.get("provenance", {}), dict):
        raise ValueError("provenance alanı bir JSON nesnesi olmalı.")
    if not isinstance(report.get("limitations", []), list):
        raise ValueError("limitations alanı liste olmalı.")
    return report


def frame(rows: list[dict]) -> pd.DataFrame:
    result = pd.DataFrame(rows)
    for name in ("event_id", "driver", "model"):
        if name in result:
            result[name] = result[name].astype(str)
    return result


def display_table(data: pd.DataFrame) -> None:
    if data.empty:
        st.info("Bu bölüm için raporda kayıt bulunmuyor.")
    else:
        st.dataframe(data, hide_index=True, width="stretch")


def numeric_chart(data: pd.DataFrame, column: str, label: str) -> None:
    if column not in data:
        return
    plot = data[["driver", column]].copy()
    plot[column] = pd.to_numeric(plot[column], errors="coerce")
    plot = plot.dropna()
    if plot.empty:
        return
    chart = alt.Chart(plot).mark_bar(color="#2865AC").encode(
        x=alt.X(f"{column}:Q", title=label),
        y=alt.Y("driver:N", sort="-x", title="Sürücü"),
        tooltip=[alt.Tooltip("driver:N", title="Sürücü"), alt.Tooltip(f"{column}:Q", title=label, format=".3f")],
    ).properties(height=max(180, 25 * len(plot)))
    st.altair_chart(chart, width="stretch")


def pace_chart(data: pd.DataFrame) -> None:
    if "predicted_pace_s" not in data:
        st.info("Bu raporda tur temposu tahmini yok.")
        return
    cols = [c for c in ("driver", "predicted_pace_s", "actual_pace_s", "lower_s", "upper_s") if c in data]
    plot = data[cols].copy()
    for col in cols[1:]:
        plot[col] = pd.to_numeric(plot[col], errors="coerce")
    plot = plot.dropna(subset=["predicted_pace_s"])
    if plot.empty:
        st.info("Gösterilebilecek sayısal tempo tahmini bulunmuyor.")
        return
    base = alt.Chart(plot).encode(y=alt.Y("driver:N", sort=alt.SortField("predicted_pace_s"), title="Sürücü"))
    points = base.mark_point(filled=True, color="#2865AC", size=80).encode(
        x=alt.X("predicted_pace_s:Q", scale=alt.Scale(zero=False), title="Tur temposu (s) · düşük daha hızlı"),
        tooltip=["driver:N", "predicted_pace_s:Q"],
    )
    layers = []
    if {"lower_s", "upper_s"}.issubset(plot.columns):
        valid = plot[plot["lower_s"].le(plot["upper_s"])].dropna(subset=["lower_s", "upper_s"])
        layers.append(alt.Chart(valid).mark_rule(color="#2865AC", strokeWidth=2).encode(
            y=alt.Y("driver:N", sort=alt.SortField("predicted_pace_s")),
            x="lower_s:Q", x2="upper_s:Q", tooltip=["driver:N", "lower_s:Q", "upper_s:Q"],
        ))
    layers.append(points)
    if "actual_pace_s" in plot:
        layers.append(base.mark_point(shape="cross", color="#B24B28", size=100).encode(x="actual_pace_s:Q", tooltip=["driver:N", "actual_pace_s:Q"]))
    st.altair_chart(alt.layer(*layers).properties(height=max(200, 30 * len(plot))), width="stretch")
    st.caption("Mavi nokta: tahmin. Turuncu artı: gerçekleşen tempo (varsa). Çizgi: raporda verilen alt/üst sınır; güven düzeyi ve yöntem için Yöntem sekmesine bakın.")


def telemetry_explorer() -> None:
    """Display separately captured historical telemetry, never fetch a session."""
    st.subheader("Seans sonrası telemetri incelemesi")
    st.caption("Bu bölüm ayrı bir tarihsel seans çıktısını okur. Yukarıdaki tahmin raporuyla aynı çalışma olmak zorunda değildir; sentetik tahminleri gerçek sonuçlara dönüştürmez.")
    path = Path(os.environ.get("RACE_TELEMETRY_PATH", str(ROOT / "reports" / "local" / "telemetry" / "summary.json")))
    if not path.exists():
        st.info("Henüz yerel telemetri çıktısı yok. README'deki telemetri komutunu çalıştırın; sayfa veri indirmez.")
        st.code("python -m f1_research.telemetry --provider openf1 --session-key 9468 --driver-numbers 1 16 --output reports/local/telemetry", language="bash")
        return
    try:
        if path.stat().st_size > MAX_REPORT_BYTES:
            raise ValueError("Telemetri özeti 25 MB sınırını aşıyor.")
        artifact = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(artifact, dict) or artifact.get("schema_version") != 1 or artifact.get("series") != "f1" or artifact.get("data_kind") != "historical" or artifact.get("analysis_kind") != "retrospective_session_analysis":
            raise ValueError("Beklenen çıktı: v1 F1 tarihsel seans analizi.")
        traces = artifact.get("telemetry", [])
        if not isinstance(traces, list) or any(not isinstance(row, dict) for row in traces):
            raise ValueError("telemetry alanı kayıt listesi olmalı.")
        for entry in traces:
            if not isinstance(entry.get("trace", []), list) or any(not isinstance(row, dict) for row in entry.get("trace", [])):
                raise ValueError("Her telemetri izi kayıt listesi olmalı.")
    except (OSError, ValueError, UnicodeError) as exc:
        st.warning(f"Telemetri özeti açılamadı: {exc}")
        return
    st.info(f"TARİHSEL SEANS ANALİZİ · {artifact.get('year', '')} {artifact.get('event', '')} · {artifact.get('session', '')}")
    st.caption(f"Kaynak: {artifact.get('source_url', 'Belirtilmedi')} · Alınma zamanı: {artifact.get('retrieved_at', 'Belirtilmedi')} · Sağlayıcı sürümü: {artifact.get('provider_version', 'Belirtilmedi')}")
    a, b = st.columns(2)
    a.metric("Seanstaki tur kaydı", artifact.get("laps", "—"))
    if artifact.get("clean_laps") is not None:
        b.metric("Temiz tur filtresini geçen", artifact["clean_laps"])
    else:
        b.metric("Zamanlama koşulunu geçen tur", artifact.get("eligible_laps", "—"))
        st.warning("Bu kaynaktaki turlar temiz tur olarak doğrulanmadı. Silinen turlar, pist durumu ve pit girişleri ayrıca kontrol edilmelidir.")
    if artifact.get("quality_scope"):
        st.write(f"**Kalite kapsamı:** {artifact['quality_scope']}")
    pace = artifact.get("pace", [])
    if isinstance(pace, list) and pace:
        st.write("Sürücü temposu (saniye)")
        display_table(pd.DataFrame(pace))
    stints = artifact.get("stints", [])
    if isinstance(stints, list) and stints:
        with st.expander("Stint özeti"):
            display_table(pd.DataFrame(stints))
            st.caption("Gözlenen tempo eğimi tek başına lastik aşınması etkisi değildir; yakıt yükü, trafik, pist ve hava koşulları da değişir.")
    if traces:
        labels = {f"{entry.get('driver', '?')} · tur {entry.get('lap', '?')} · {i + 1}": entry for i, entry in enumerate(traces)}
        choices = st.multiselect("Karşılaştırılacak turlar (en fazla 2)", list(labels), default=list(labels)[:2], max_selections=2)
        combined = []
        for label in choices:
            for point in labels[label].get("trace", []):
                combined.append({**point, "tur": label})
        plot = pd.DataFrame(combined)
        if not plot.empty and "distance_m" in plot:
            plot["distance_m"] = pd.to_numeric(plot["distance_m"], errors="coerce")
            channels = [("speed_kmh", "Hız (km/sa)"), ("throttle_pct", "Gaz (%)"), ("brake", "Fren (0 = kapalı, 1 = açık)")]
            for channel, title in channels:
                if channel not in plot:
                    continue
                plot[channel] = pd.to_numeric(plot[channel], errors="coerce")
                valid = plot.dropna(subset=["distance_m", channel])
                if valid.empty:
                    continue
                chart = alt.Chart(valid).mark_line(interpolate="step-after" if channel == "brake" else "linear").encode(
                    x=alt.X("distance_m:Q", title="Tur içi türetilmiş mesafe (m)"),
                    y=alt.Y(f"{channel}:Q", title=title),
                    color=alt.Color("tur:N", title="Seçili tur"),
                    strokeDash=alt.StrokeDash("tur:N", legend=None),
                    tooltip=["tur:N", alt.Tooltip("distance_m:Q", format=".1f"), alt.Tooltip(f"{channel}:Q", format=".1f")],
                ).properties(height=190)
                st.altair_chart(chart, width="stretch")
            st.caption("Mesafe hız/zamandan türetilir; sağlayıcıya göre ilk örneğe göreli başlayabilir. Tam konumsal eşleştirme veya kesin tur farkı ölçümü değildir. Fren kanalı açık/kapalı bilgisidir, fren basıncı değildir. Çizgiler farklı turların ölçümlerini mesafe ekseninde karşılaştırır.")
            with st.expander("Seçili turların örnekleme ve kalite bilgileri"):
                st.json([{key: value for key, value in labels[label].items() if key != "trace"} for label in choices])
        else:
            st.info("Mesafe tabanlı grafik için bir tur seçin; seçili izlerde distance_m alanı bulunmalı.")
    else:
        st.info("Bu seans özetinde telemetri izi yok; mevcut tur/tempo kayıtları yukarıda gösterilir.")
    for missing in artifact.get("unavailable", []):
        st.warning(f"Erişilemeyen veri: {missing}")
    with st.expander("Telemetri analizinin sınırları"):
        for limitation in artifact.get("limitations", []):
            st.write(f"• {limitation}")



def replay_explorer():
    """Display a separate historical replay without relabeling pre-race forecasts."""
    folder = ROOT / "reports" / "local" / ("replay" if SERIES == "f1" else "lap-replay")
    if SERIES == "f1":
        choices = [p for p in (ROOT / "reports/local/replay-latest", folder) if (p / "report.json").exists()]
        if choices:
            folder = st.selectbox("Replay çalışması", choices, format_func=lambda p: p.name)
    path = folder / "report.json"
    if not path.exists():
        st.info("Tur bazlı replay henüz üretilmedi. README içindeki replay komutunu çalıştırın.")
        return
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("series") != SERIES:
            raise ValueError("Replay serisi uyuşmuyor")
        st.subheader("Turdan tura tahmin · geçmiş yarış replay")
        session = report.get("test_session", {})
        if session:
            st.caption(f"Replay testi: {session.get('location', session.get('country_name', ''))} · {session.get('date_start', '')}")
        st.warning("Bu bir geçmiş yarış simülasyonudur. Canlı bağlantı ve gerçek veri geliş gecikmesi doğrulanmış değildir.")
        st.dataframe(pd.DataFrame(report["summary"]), hide_index=True, width="stretch")
        csv = folder / "predictions.csv"
        if csv.exists():
            predictions = pd.read_csv(csv)
            identity = "driver_number" if SERIES == "f1" else "driver"
            driver = st.selectbox("Replay sürücüsü", predictions[identity].drop_duplicates().tolist())
            selected = predictions[predictions[identity] == driver]
            if SERIES == "f1":
                chart = selected.melt(id_vars=["lap_number"], value_vars=["actual", "ridge", "last_lap", "recent_median"], var_name="seri", value_name="saniye")
                axis = "lap_number"
            else:
                forecast = selected.rename(columns={"model": "seri", "prediction_s": "saniye"})
                actual = selected.drop_duplicates("target_lap").assign(seri="actual").rename(columns={"actual_s": "saniye"})
                chart = pd.concat([forecast, actual], ignore_index=True)
                axis = "target_lap"
            st.altair_chart(alt.Chart(chart).mark_line(point=True).encode(
                x=alt.X(f"{axis}:Q", title="Tahmin edilen tur"),
                y=alt.Y("saniye:Q", title="Tur süresi (s)", scale=alt.Scale(zero=False)),
                color="seri:N", tooltip=[f"{axis}:Q", "seri:N", "saniye:Q"]), width="stretch")
            st.dataframe(selected, hide_index=True, width="stretch")
        with st.expander("Tuning, zaman kontrolü ve sınırlamalar"):
            st.json({key: report[key] for key in ("selected_alpha", "cutoff_lap", "tuning", "cutoff_checks", "limitations") if key in report})
    except (ValueError, KeyError, OSError) as exc:
        st.error(f"Replay raporu açılamadı: {exc}")


def main() -> None:
    st.set_page_config(page_title=TITLE, page_icon="🏁", layout="wide")
    st.title(TITLE)
    st.caption("Tahmin · kanıt · veri kaynağı — aynı çalışma üzerinden.")
    with st.sidebar:
        st.header("Çalışma seçimi")
        paths = [p for p in sorted((ROOT / "reports").rglob("report.json")) if "replay" not in str(p)] if (ROOT / "reports").exists() else []
        default = Path(os.environ.get("RACE_REPORT_PATH", str(ROOT / "reports" / "demo" / "report.json")))
        choices = list(dict.fromkeys([default, *paths]))
        selected_path = st.selectbox("Yerel rapor", choices, format_func=lambda p: str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else p.name)
        uploaded = st.file_uploader("Veya rapor yükleyin", type=["json"], help="Yüklenen JSON yalnız görüntülenir; kod çalıştırılmaz.")
        st.caption("Sayfa açılırken model eğitilmez, veri indirilmez. Seçilen raporun kayıtlı sonuçları gösterilir.")
    try:
        raw = uploaded.getvalue() if uploaded is not None else selected_path.read_bytes()
        report = read_report(raw)
    except FileNotFoundError:
        st.info("Henüz görüntülenecek rapor yok.")
        st.markdown("README'deki demo komutunu çalıştırın; oluşan **reports/demo/report.json** burada açılır. Hazır bir raporunuz varsa kenar çubuğundan yükleyebilirsiniz.")
        st.code(f"{ROOT / 'reports' / 'demo' / 'report.json'}", language=None)
        return
    except (ValueError, UnicodeError, OSError) as exc:
        st.error(f"Rapor açılamadı: {exc}")
        return

    if report["data_kind"] == "synthetic":
        st.warning("SENTETİK DEMO — Bu veriler iş akışını göstermek için üretilmiştir. Gerçek yarış başarısı veya tarihsel performans kanıtı değildir.")
    else:
        st.info("TARİHSEL VERİ — Kaynak ve değerlendirme kapsamını Veri sekmesinden kontrol edin. Tarihsel veri etiketi tek başına doğrulanmış tahmin başarısı anlamına gelmez.")
    st.caption(f"Çalışma: {report.get('run_id', 'Belirtilmedi')} · Tahmin kesimi: {report.get('forecast_origin', 'Belirtilmedi')} · Rapor v1")
    predictions = frame(report.get("predictions", []))
    metrics = frame(report.get("metrics", []))
    summary = frame(report.get("summary", []))
    events = predictions["event_id"].drop_duplicates().tolist() if not predictions.empty else []
    selected = predictions
    if events:
        a, b = st.columns(2)
        event = a.selectbox("Yarış / seans", events)
        by_event = predictions[predictions["event_id"] == event]
        model = b.selectbox("Model", by_event["model"].drop_duplicates().tolist())
        selected = by_event[by_event["model"] == model]
    else:
        event, model = None, None
    overview, forecast, evidence, data, method = st.tabs(["Genel bakış", "Tahmin", "Kanıt / karşılaştırma", "Tempo ve veri", "Yöntem / sınırlar"])

    with overview:
        st.subheader("Araştırmanın kapsamı")
        a, b, c = st.columns(3)
        a.metric("Raporlanan yarış / seans", len(events))
        b.metric("Benzersiz sürücü", predictions["driver"].nunique() if not predictions.empty else 0)
        c.metric("Karşılaştırılan model", predictions["model"].nunique() if not predictions.empty else 0)
        st.markdown("**Önce kanıtı okuyun.** Yarış seçerek model çıktılarını inceleyin; ardından aynı yarıştaki referans modellerle karşılaştırın. Verinin eksik olduğu alanlar ve yöntem sınırları raporla birlikte gösterilir.")
        st.subheader("Raporun özet ölçümleri")
        display_table(summary)
        st.caption("MAE: ortalama mutlak hata. Log loss / Brier: olasılık tahmininin hatası. Winner accuracy: ilk seçimin kazanan olması. Coverage: gerçekleşen temponun tahmin aralığı içinde kalma oranı.")
        st.caption("Özetler rapordan olduğu gibi okunur. Bir sürücü satırı bağımsız bir yarış örneği değildir; model seçimini tek bir iyi yarışa dayandırmayın.")
        if event is not None:
            st.write(f"Seçili yarış: **{event}** · Model: **{model}**")

    with forecast:
        st.subheader("Seçili yarışın tahminleri")
        if selected.empty:
            st.info("Rapor tahmin kaydı içermiyor.")
        else:
            if SERIES == "f1":
                numeric_chart(selected, "win_probability", "Kazanma olasılığı (0–1)")
                if "win_probability" in selected:
                    probabilities = pd.to_numeric(selected["win_probability"], errors="coerce")
                    if probabilities.notna().all():
                        st.caption(f"Bu model / yarış için raporlanan olasılık toplamı: {probabilities.sum():.4f}")
                        if not probabilities.between(0, 1).all() or abs(probabilities.sum() - 1) > 0.01:
                            st.warning("Olasılıklar ortak bir yarış dağılımı koşulunu sağlamıyor. Kesin kazanma dağılımı olarak yorumlamayın.")
            else:
                pace_chart(selected)
            order = "predicted_position" if SERIES == "f1" else "predicted_pace_s"
            display_table(selected.sort_values(order) if order in selected else selected)
            st.download_button("Seçili tahminleri CSV indir", selected.to_csv(index=False).encode("utf-8-sig"), file_name=f"{SERIES}-predictions.csv", mime="text/csv")

    with evidence:
        st.subheader("Aynı yarışta model karşılaştırması")
        event_metrics = metrics[metrics["event_id"] == event] if event and "event_id" in metrics else metrics
        display_table(event_metrics)
        if not event_metrics.empty and "model" in event_metrics:
            numeric = [name for name in ("winner_log_loss", "winner_brier", "position_mae", "winner_accuracy", "podium_recall", "mae_s", "coverage_80", "mean_interval_width_s", "median_ae_s", "mae") if name in event_metrics]
            if numeric:
                metric = st.selectbox("Karşılaştırma ölçütü", numeric)
                chart = alt.Chart(event_metrics).mark_bar(color="#2865AC").encode(x=alt.X(f"{metric}:Q", title=metric), y=alt.Y("model:N", title="Model"), tooltip=["model:N", alt.Tooltip(f"{metric}:Q", format=".4f")])
                st.altair_chart(chart, width="stretch")
        st.caption("MAE, log loss ve Brier için düşük değer daha iyidir. Korelasyon ve kapsama gibi ölçütler farklı yorumlanır; sayısal ölçekleri birbirleriyle karşılaştırmayın.")
        with st.expander("Tüm yarışların ölçümleri"):
            display_table(metrics)
        if report.get("paired_logloss_vs_qualifying"):
            st.subheader("Sıralama referansına göre olasılık hatası farkı")
            comparison = [{"model": name, "ortalama_fark": value["mean"],
                           "alt_95": value["interval_95"][0], "ust_95": value["interval_95"][1]}
                          for name, value in report["paired_logloss_vs_qualifying"].items()]
            display_table(pd.DataFrame(comparison))
            st.caption("Negatif fark aday lehine. Aralık, aynı yarışları eşleştirerek yeniden örnekler; ardışık yarış bağımlılığını hesaba katmaz.")
        if report.get("reliability_bins"):
            st.subheader("Olasılıklar gerçekleşme oranıyla uyuşuyor mu?")
            calibration = pd.DataFrame(report["reliability_bins"])
            calibration = calibration[calibration.model == model]
            chart = alt.Chart(calibration).mark_circle(size=75).encode(
                x=alt.X("mean_probability:Q", title="Tahmin edilen kazanma olasılığı", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("observed_win_rate:Q", title="Gözlenen kazanma oranı", scale=alt.Scale(domain=[0, 1])),
                tooltip=["count:Q", "mean_probability:Q", "observed_win_rate:Q"])
            reference = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(strokeDash=[4, 4], color="gray").encode(x="x:Q", y="y:Q")
            st.altair_chart(reference + chart, width="stretch")
            st.caption("Kesikli çizgi tam uyumu gösterir. Az gözlemli olasılık aralıkları güvenilir bir calibration kanıtı değildir.")
        st.markdown("**Değerlendirme kontrolü:** Eğitim yarışları testten önce mi? Aynı yarışın sürücüleri aynı test bölümünde mi? Referans ve aday aynı bilgiye erişiyor mu? Bu ekran doğrulama protokolünün yerini tutmaz.")

    with data:
        replay_explorer()
        st.subheader("Veri kapsamı ve kalite")
        if not predictions.empty:
            coverage = predictions.groupby(["event_id", "model"], sort=False).agg(surucu=("driver", "nunique"), satir=("driver", "size")).reset_index()
            display_table(coverage)
            missing = predictions.isna().sum()
            missing = missing[missing.gt(0)].rename_axis("alan").reset_index(name="eksik_satir")
            if missing.empty:
                st.caption("Tahmin tablosunun mevcut sütunlarında boş hücre yok. Bu, kaynak veri kapsamının tam olduğu anlamına gelmez.")
            else:
                st.write("Tahmin tablosundaki eksik değerler")
                display_table(missing)
            duplicates = int(predictions.duplicated(["event_id", "driver", "model"]).sum())
            if duplicates:
                st.warning(f"Aynı yarış / sürücü / model için {duplicates} yinelenen kayıt var.")
        st.subheader("Kaynak izi")
        st.json(report.get("provenance", {}), expanded=False)
        if SERIES == "f1":
            st.info("Bu sayfadaki sonuç tablosu sensör telemetrisi değildir. Telemetri analizi için ayrıca seans, tur, kanal ve kaynak kaydı bulunan bir çıktı gerekir.")
            telemetry_explorer()
        else:
            st.info("Tur / sektör zamanı, motosikletin sensör telemetrisi değildir. Bu arayüz raporda bulunmayan hız, gaz veya fren kanalları üretmez.")
            if not selected.empty:
                pace_chart(selected)

    with method:
        st.subheader("Bu çıktıyı nasıl okumalı?")
        st.markdown("Tahminler seçili çalışmanın kayıtlı çıktılarıdır. Gerçekleşen sonuçlar yalnız karşılaştırma içindir; tahmin anında bilinemeyen bilgilerin eğitim özelliklerine sızmadığı ayrıca test edilmelidir.")
        st.markdown("Özellik öneminin yüksek olması nedensel etkiyi kanıtlamaz. Bu rapor açıklama veya özellik çıkarma deneyi içermiyorsa, ekranda bir açıklama uydurulmaz.")
        extra = {key: report[key] for key in ("methodology", "interval_method", "feature_importance", "ablations", "split_manifest") if key in report}
        if extra:
            st.json(extra, expanded=False)
        st.subheader("Çalışmanın sınırlamaları")
        limitations = report.get("limitations", [])
        if limitations:
            for limitation in limitations:
                st.write(f"• {limitation}")
        else:
            st.warning("Rapor sınırlama belirtmiyor. Bu, modelin sınırsız veya doğrulanmış olduğu anlamına gelmez.")
        st.download_button("Çalışma raporunu indir", raw, file_name=f"{SERIES}-report.json", mime="application/json")


if __name__ == "__main__":
    main()
