from f1_research.race_control import TrackState, reduce_race_control


def test_race_control_safety_car_red_restart_finish():
    state = reduce_race_control(TrackState(), {"date": "2026-01-01T00:00:00Z", "message": "SAFETY CAR DEPLOYED"})
    assert state.state == "SC"
    state = reduce_race_control(state, {"date": "2026-01-01T00:01:00Z", "flag": "RED", "message": "RED FLAG"})
    assert state.state == "RED"
    state = reduce_race_control(state, {"date": "2026-01-01T00:02:00Z", "flag": "GREEN"})
    assert state.state == "RESTART"
    state = reduce_race_control(state, {"date": "2026-01-01T00:03:00Z", "flag": "CHEQUERED"})
    assert state.state == "FINISHED"


def test_unknown_message_does_not_invent_state():
    current = TrackState("GREEN")
    assert reduce_race_control(current, {"message": "CAR 1 TRACK LIMITS"}) == current
