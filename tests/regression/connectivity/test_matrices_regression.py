"""Gate-tier regression: connectivity matrix math (deterministic, hermetic)."""

from __future__ import annotations

import pytest

from neurofaune.network.matrices import fisher_z_transform, spearman_matrix
from tests.regression import fixtures
from tests.regression._equivalence import assert_array_matches_golden

pytestmark = pytest.mark.regression


def test_spearman_matrix_preserved():
    data = fixtures.roi_subject_matrix()
    result = spearman_matrix(data)
    assert_array_matches_golden(result, "connectivity_spearman_matrix")


def test_fisher_z_transform_preserved():
    r = fixtures.correlation_vector()
    result = fisher_z_transform(r)
    assert_array_matches_golden(result, "connectivity_fisher_z")
