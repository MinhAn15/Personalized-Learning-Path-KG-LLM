import pytest

from backend.src.path_generator import calculate_dynamic_weights


class DummyProfile:
    pass


def test_calculate_dynamic_weights_no_student():
    dw = calculate_dynamic_weights(None)
    assert isinstance(dw, dict)
    assert 'time_estimate' in dw
    assert dw['time_estimate'] == 1.0


def test_calculate_dynamic_weights_with_profile(monkeypatch):
    # Create a fake profile with performance_details list
    fake_profile = {
        'performance_details': ['n1:80:30', 'n2:90:45', 'n3:100:60']
    }

    def fake_load_student_profile(student_id):
        return fake_profile

    monkeypatch.setattr('backend.src.path_generator.load_student_profile', fake_load_student_profile)

    dw = calculate_dynamic_weights('student_123')
    assert isinstance(dw, dict)
    assert 'time_estimate' in dw
    # avg time = (30+45+60)/3 = 45 -> scalar = 60/45 = 1.333 -> clamped -> 1.333
    assert abs(dw['time_estimate'] - (60.0/45.0)) < 0.01
