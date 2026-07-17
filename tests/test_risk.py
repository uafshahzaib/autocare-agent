"""Unit tests for the risk-scoring / triage tool (pure Python, no model calls)."""
from app.tools import _assess_risk


def test_risk_critical_when_overdue_and_red_light():
    result = _assess_risk(20000, 0, 15000, active_warning_light="oil")
    assert "CRITICAL" in result


def test_risk_high_when_red_light_only():
    result = _assess_risk(5000, 0, 15000, active_warning_light="brake")
    assert "HIGH" in result


def test_risk_high_when_overdue_only():
    result = _assess_risk(20000, 0, 15000, active_warning_light="none")
    assert "HIGH" in result


def test_risk_medium_when_amber_light():
    result = _assess_risk(5000, 0, 15000, active_warning_light="tire")
    assert "MEDIUM" in result


def test_risk_low_when_nothing_wrong():
    result = _assess_risk(5000, 0, 15000, active_warning_light="none")
    assert "LOW" in result
