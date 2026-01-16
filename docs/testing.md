# Testing

## Local test run

Install the dev dependencies and run the test suite:

```bash
python -m pip install -e ".[dev]"
pytest -q
```

## Makefile helper

Alternatively, use the Makefile shortcut:

```bash
make test
```

## Smoke run

Run the end-to-end smoke script:

```bash
python scripts/smoke_run.py
```
