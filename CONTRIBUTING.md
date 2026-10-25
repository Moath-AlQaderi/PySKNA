# Contributing to PySKNA

Bug reports, validated datasets, documentation improvements, and new analysis methods are welcome.

1. Create a branch from `main`.
2. Install development dependencies with `python -m pip install -e ".[dev]"`.
3. Add tests for behavioral changes.
4. Run `pytest` and `ruff check .`.
5. Open a pull request explaining the physiological method and citing its source.

Do not commit identifiable participant data. Any clinical method added to PySKNA should document its acquisition assumptions, units, sampling rate, filter behavior, and validation status.

