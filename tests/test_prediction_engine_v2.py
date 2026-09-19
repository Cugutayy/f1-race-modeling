from f1_research.prediction_engine_v2 import assess_quality, prediction_envelope


def test_prediction_envelope_fuses_pace_and_race_without_inventing_values():
    state={"session_key":1,"current_lap":20,"updated_at":"2026-01-01T00:00:00Z","state_age_s":2.0,
           "drivers":[{"driver_number":1,"acronym":"AAA","position":1,"last_lap_s":90.0}]}
    race=[{"driver_number":1,"label":"AAA","expected_position":1.4,"win_probability":.6,
           "podium_probability":.9,"top10_probability":1.0,"dnf_probability":.03}]
    pace=[{"driver_number":1,"predicted_green_lap_s":89.8,"recent_median_5_s":90.1,
           "green_lap_lower_s":88.5,"green_lap_upper_s":91.1,"p_green":.96,"p_pit":.02,"p_neutralized":.02}]
    out=prediction_envelope(state=state,race_predictions=race,pace_predictions=pace)
    assert out["schema_version"]==2
    assert out["quality"]["degraded"] is False
    assert out["drivers"][0]["pace_delta_vs_recent_median_s"] == -0.29999999999999716
    assert out["drivers"][0]["win_probability"] == .6

def test_quality_fails_visible_when_strict_pace_is_missing():
    q=assess_quality({"drivers":[{"driver_number":1,"pace_laps_s":[90,91,90]}]},[])
    assert q.degraded
    assert "strict_pace_unavailable" in q.reasons
