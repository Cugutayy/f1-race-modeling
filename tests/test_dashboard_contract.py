from f1_research.dashboard_contract import build_dashboard_payload


def test_dashboard_keeps_observed_and_modelled_values_separate():
    state = {"session_key": 1, "current_lap": 5, "drivers": [
        {"driver_number": 1, "position": 1, "gap_to_leader_s": 0.0, "compound": "MEDIUM", "tyre_age": 5}
    ]}
    prediction = {"prediction_id": "p1", "report": {"predictions": [
        {"driver_number": 1, "win_probability": 0.7, "expected_position": 1.4}
    ]}}
    payload = build_dashboard_payload(state=state, prediction=prediction, quality_status="LIVE")
    row = payload["leaderboard"][0]
    assert row["position"] == 1
    assert row["win_probability"] == 0.7
    assert payload["prediction_id"] == "p1"
