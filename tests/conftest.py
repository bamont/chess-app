import importlib.util

GUI_TESTS = ["test_gui_assets.py", "test_gui_geometry.py"]
PYGAME_MISSING = importlib.util.find_spec("pygame") is None

collect_ignore = GUI_TESTS if PYGAME_MISSING else []


def pytest_terminal_summary(terminalreporter) -> None:
    if PYGAME_MISSING:
        terminalreporter.write_line("pygame is not installed: GUI tests were not collected (uv sync --extra gui)")