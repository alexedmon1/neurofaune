# Regression (behavior-preservation) tests

These tests enforce the **boundary** of an implementation: given a frozen
input, a candidate must reproduce a frozen output within tolerance. They are
the half of the gate that the unit tests don't cover — *did my change keep the
results the same?* — and they are what lets a refactor (e.g. the
`MODERNIZATION_PLAN.md` class-based rewrites) be promoted with confidence.

## Layout

```
tests/regression/
├── _equivalence.py   # helpers: fixture_path, golden_path, assert_array_matches_golden
├── fixtures/         # frozen inputs (small, committed)
├── golden/           # expected outputs (committed)
└── test_*.py         # @pytest.mark.regression tests
```

## Writing a regression test for a real change

1. Capture a **small** representative input into `fixtures/`.
2. Run the *current* (trusted) implementation once to mint the golden:
   ```bash
   NEUROFAUNE_UPDATE_GOLDEN=1 uv run pytest -m regression
   ```
   Review and commit the new `golden/*.npy` in its own commit.
3. Write the test so the **candidate** implementation runs against the fixture
   and is compared to the golden:
   ```python
   import pytest
   from tests.regression._equivalence import fixture_path, assert_array_matches_golden

   pytestmark = pytest.mark.regression

   def test_fc_matrix_preserved():
       result = compute_fc_matrix(fixture_path("ts_small.npy"))   # candidate
       assert_array_matches_golden(result, "fc_matrix")
   ```
4. `make check` (or `make regression`) now fails if any candidate drifts.

## The loop

When two implementations compete (old vs refactor), check each out in its own
worktree (`~/sandbox/.worktrees/...`), run `make check` in each, and compare
against the *same* committed golden. Only a green candidate gets tagged and
pinned in research. See `plans/main-plan.md`.
