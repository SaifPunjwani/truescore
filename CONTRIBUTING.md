# Contributing

## Setup

```sh
git clone https://github.com/SaifPunjwani/truescore
cd truescore
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
python examples/generate_sample_data.py   # the examples and CLI tests read these files
```

## What CI runs

```sh
pytest -q
ruff check src tests examples
ruff format --check src tests examples
mypy
```

All four have to pass. There are no known-flaky tests. If a seeded test fails
intermittently, that's a bug, please report it.

## What a change needs

**New estimator**: a derivation in `docs/methods/` with the algebra written out, and a
coverage simulation showing the interval covers at its nominal rate. Also say what it
assumes and where it breaks.

**New diagnostic**: a simulation showing its false-positive rate when nothing is wrong. A
threshold without a measured error rate isn't usable.

**Any public function**: given any input, return a finite result or raise `ValueError`.
Never a silent NaN. The fuzz suite in `tests/test_properties.py` enforces this.

**Docs**: cite the pytest node id of the test that enforces each claim.

## Style

Whatever `ruff` and `mypy --strict` accept. Two extra rules: comments explain constraints
rather than restating the code, and every number in the docs is either derived on the page
or printed by something in the repo.

## Reporting a statistical bug

These are the most useful reports and the easiest to under-describe. Include the inputs (a
seed and array shapes is usually enough), what you got, and what you expected.

If you think an interval is undercovering, a short loop showing the observed coverage rate
is the most useful thing you can attach. That's how the three bugs fixed before the first
release were found.
