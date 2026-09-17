"""Offline UI contract tests, independent of model training and data providers."""

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app" / "dashboard.py"
SERIES = "f1"


@pytest.fixture
def report_path(tmp_path, monkeypatch):
    predictions = []
    metrics = []
    for event in ("demo-01", "demo-02"):
        for model in ("grid_baseline", "research_model"):
            for position, driver in enumerate(("Demo A", "Demo B"), start=1):
                predictions.append({
                    "event_id": event, "driver": driver, "model": model,
                    "actual_position": position, "predicted_position": position,
                    "win_probability": 0.6 if position == 1 else 0.4,
                    "actual_pace_s": 90.0 + position,
                    "predicted_pace_s": 90.1 + position,
                    "lower_s": 89.1 + position, "upper_s": 91.1 + position,
                })
            metrics.append({"event_id": event, "model": model, "mae": 0.1})
    report = {
        "schema_version": 1, "series": SERIES, "data_kind": "synthetic",
        "run_id": "test-only", "forecast_origin": "2025-01-01T00:00:00Z",
        "summary": [{"model": "grid_baseline", "mae": 0.1}],
        "metrics": metrics, "predictions": predictions,
        "provenance": {"source": "synthetic test fixture"},
        "limitations": ["Fixture values are not a measured racing result."],
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setenv("RACE_REPORT_PATH", str(path))
    monkeypatch.setenv("RACE_TELEMETRY_PATH", str(tmp_path / "no-telemetry.json"))
    return path


def test_offline_report_and_event_model_filters(report_path):
    app = AppTest.from_file(str(APP)).run(timeout=20)
    assert not app.exception
    assert len(app.tabs) == 5
    assert any("SENTETİK DEMO" in item.value for item in app.warning)
    assert [item.value for item in app.metric] == ["2", "2", "2"]
    selectors = {widget.label: widget for widget in app.selectbox}
    selectors["Yarış / seans"].set_value("demo-02").run()
    selectors = {widget.label: widget for widget in app.selectbox}
    selectors["Model"].set_value("research_model").run()
    assert not app.exception
    # A displayed prediction table must match BOTH controls, not just the event.
    tables = [widget.value for widget in app.dataframe]
    selected = next(table for table in tables if "win_probability" in table.columns)
    assert set(selected["event_id"]) == {"demo-02"}
    assert set(selected["model"]) == {"research_model"}


def test_missing_report_explains_how_to_start(tmp_path, monkeypatch):
    monkeypatch.setenv("RACE_REPORT_PATH", str(tmp_path / "missing.json"))
    app = AppTest.from_file(str(APP)).run(timeout=20)
    assert not app.exception
    assert any("Henüz görüntülenecek rapor yok" in item.value for item in app.info)


@pytest.mark.parametrize("invalid", ["not json", '{"schema_version": 99}', '[]'])
def test_invalid_report_is_a_friendly_error(report_path, invalid):
    report_path.write_text(invalid, encoding="utf-8")
    app = AppTest.from_file(str(APP)).run(timeout=20)
    assert not app.exception
    assert any("Rapor açılamadı" in item.value for item in app.error)


def test_wrong_series_cannot_be_mislabelled(report_path):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["series"] = "worldsbk" if SERIES == "f1" else "f1"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    app = AppTest.from_file(str(APP)).run(timeout=20)
    assert not app.exception
    assert any("yalnız" in item.value for item in app.error)


def test_retrospective_telemetry_stays_separate_from_synthetic_forecast(report_path, monkeypatch):
    telemetry = {
        "schema_version": 1, "series": "f1", "data_kind": "historical",
        "analysis_kind": "retrospective_session_analysis", "event": "Fixture GP",
        "year": 2025, "session": "Q", "laps": 12, "clean_laps": None,
        "eligible_laps": 8, "quality_scope": "Timing only; clean lap status unverified.",
        "telemetry": [
            {"driver": driver, "lap": 3, "trace": [
                {"distance_m": 0, "speed_kmh": 200, "throttle_pct": 100, "brake": False},
                {"distance_m": 50, "speed_kmh": 180, "throttle_pct": 20, "brake": True},
            ]} for driver in ("AAA", "BBB")
        ],
        "limitations": ["Test-only trace."], "unavailable": [],
    }
    path = report_path.parent / "telemetry.json"
    path.write_text(json.dumps(telemetry), encoding="utf-8")
    monkeypatch.setenv("RACE_TELEMETRY_PATH", str(path))
    app = AppTest.from_file(str(APP)).run(timeout=20)
    assert not app.exception
    assert any("SENTETİK DEMO" in item.value for item in app.warning)
    assert any("TARİHSEL SEANS ANALİZİ" in item.value for item in app.info)
    assert any("temiz tur olarak doğrulanmadı" in item.value for item in app.warning)
    assert next(item.value for item in app.metric if item.label == "Zamanlama koşulunu geçen tur") == "8"
    assert len(app.multiselect[0].value) == 2
    app.multiselect[0].set_value([]).run()
    assert not app.exception
    assert any("bir tur seçin" in item.value for item in app.info)
