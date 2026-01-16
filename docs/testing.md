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

## CLI usage

Run a single turn (defaults to session `demo-1`):

```bash
python -m spectator run --text "Hello"
```

Start the interactive REPL (type `/exit` or EOF to quit):

```bash
python -m spectator repl --session demo-1
```

By default, data is stored under `./data/` (checkpoints, traces, sandbox).
