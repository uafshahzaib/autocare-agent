"""Unit tests for the deterministic tools (no LLM/network required)."""
from app.tools import _lookup_warning_light, _service_interval_check


def test_service_interval_not_due_yet():
    result = _service_interval_check(current_mileage_km=10000, last_service_mileage_km=5000, service_interval_km=15000)
    assert "No service due yet" in result
    assert "10000" in result or "10,000 km remain" not in result  # sanity: no crash on formatting


def test_service_interval_due_soon():
    result = _service_interval_check(current_mileage_km=14500, last_service_mileage_km=0, service_interval_km=15000)
    assert "due soon" in result


def test_service_interval_overdue():
    result = _service_interval_check(current_mileage_km=20000, last_service_mileage_km=0, service_interval_km=15000)
    assert "OVERDUE" in result


def test_service_interval_invalid_input():
    result = _service_interval_check(current_mileage_km=100, last_service_mileage_km=5000, service_interval_km=15000)
    assert "cannot be lower" in result


def test_warning_light_known():
    result = _lookup_warning_light("engine")
    assert "amber" in result
    assert "misfire" in result


def test_warning_light_case_insensitive_and_fuzzy():
    result = _lookup_warning_light("OIL")
    assert "red" in result
    assert "Stop the vehicle" in result


def test_warning_light_unknown():
    result = _lookup_warning_light("windshield-wiper")
    assert "Unknown warning light" in result
