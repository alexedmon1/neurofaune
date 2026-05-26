"""Integration-tier template: end-to-end preprocessing vs derived-metric golden.

Runs only via `make integration` (marker: integration), locally or on HPC where
ANTs/FSL are installed; it is NOT part of the fast PR gate. Because ANTs/FSL are
not bit-reproducible, we compare robust derived metrics (Dice, correlation,
intensity summary) within loose tolerance — never raw voxels.

This is a skipping template: wire it to a real builder + a tiny downsampled-real
fixture, then remove the skip. It skips cleanly when tools are absent, so it
never produces a false failure.
"""

from __future__ import annotations

import shutil

import pytest

pytestmark = pytest.mark.integration

requires_ants = pytest.mark.skipif(
    shutil.which("antsRegistration") is None,
    reason="ANTs not installed; integration tier runs locally / on HPC",
)


@requires_ants
def test_anat_registration_derived_metrics(tmp_path):
    pytest.skip(
        "TEMPLATE: run a real ANTs registration on a tiny downsampled fixture, "
        "then compare derived metrics to golden. Pattern:\n"
        "  from tests.regression._derived import dice, summary, assert_metrics_close\n"
        "  out_mask, out_map = run_anat_registration(fixtures.anat_tiny(), tmp_path)\n"
        "  metrics = {\n"
        "      'dice_brainmask': dice(out_mask, golden_mask),\n"
        "      **{f'fa_{k}': v for k, v in summary(out_map).items()},\n"
        "  }\n"
        "  assert_metrics_close(metrics, 'anat_registration')"
    )
