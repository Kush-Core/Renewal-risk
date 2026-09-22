"""The split windows respect label resolution and the horizon."""
from renewal_risk.split import window_problems


def test_configured_windows_are_safe():
    assert window_problems() == []


def test_detects_embargo_violation(monkeypatch):
    from renewal_risk import split
    monkeypatch.setitem(split.WINDOWS, "calib", ("2016-03-01", "2016-06-30"))
    assert any("calib labels resolve" in p for p in split.window_problems())


def test_detects_short_horizon():
    assert any("outcomes for test end" in p for p in window_problems("2016-12-31"))


def test_detects_horizon_training_overlap(monkeypatch):
    from renewal_risk import split
    monkeypatch.setitem(split.WINDOWS, "horizon_train", ("2015-07-01", "2016-02-29"))
    assert any("horizon_train outcomes resolve" in p for p in split.window_problems())