from tools.timing import best_of, calibrate


def test_calibrate_returns_a_positive_duration():
    assert calibrate(repeat=1) > 0


def test_best_of_runs_the_function_every_time():
    calls = []
    duration = best_of(lambda: calls.append(1), repeat=4)
    assert len(calls) == 4
    assert duration >= 0