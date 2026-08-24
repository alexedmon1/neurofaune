"""The check that stops code from naming a tpl->SIGMA transform.

The bug this guards against is not that the path is ugly -- it is that a wrong
path here does not raise. The chain is built from Paths that do not exist, the
caller returns None, and the run finishes green with the SIGMA-space outputs
missing. Twice on the cuprizone study that was found by reading a log.
"""
from pathlib import Path

import pytest

from neurofaune.qa import find_hardcoded_sigma_paths, scan_source
from neurofaune.qa.path_hygiene import OPT_OUT

# The exact expression qc_anat_registration.py used, which resolved nothing after
# a rebuild replaced the hand-made symlink farm with ANTs' real output names.
REAL_BUG = '''
from pathlib import Path
def warp(ses):
    tpl_dir = Path("templates") / ses / "transforms"
    return [
        tpl_dir / "tpl-to-SIGMA_1Warp.nii.gz",
        tpl_dir / "tpl-to-SIGMA_0GenericAffine.mat",
    ]
'''


def test_catches_the_bug_that_motivated_it():
    found = scan_source(REAL_BUG, "qc_anat_registration.py")
    assert [f.literal for f in found] == [
        "tpl-to-SIGMA_1Warp.nii.gz",
        "tpl-to-SIGMA_0GenericAffine.mat",
    ]
    assert [f.line for f in found] == [6, 7]


@pytest.mark.parametrize("literal", [
    "tpl-to-SIGMA_0GenericAffine.mat",
    "tpl-to-SIGMA_1Warp.nii.gz",
    "tpl-to-SIGMA_1InverseWarp.nii.gz",
    # the study-prefixed spelling ANTs actually produced
    "tpl-CPZp60_to-SIGMA_0GenericAffine.mat",
    "tpl-CPZp120_to-SIGMA_1Warp.nii.gz",
])
def test_flags_every_spelling(literal):
    assert scan_source(f'x = "{literal}"')


def test_flags_the_fragment_inside_an_fstring():
    """f"tpl-{cohort}_to-SIGMA_1Warp.nii.gz" is the same bug, one interpolation on."""
    found = scan_source('c = "p60"\nx = f"tpl-{c}_to-SIGMA_1Warp.nii.gz"')
    assert len(found) == 1


def test_does_not_flag_a_per_session_affine():
    """Modality->template transforms are named by us and are not ambiguous.

    Flagging these would make the check noise, and a noisy check gets disabled.
    """
    src = 'x = "sub-1X_ses-1_T2w_to_template_0GenericAffine.mat"\ny = "FA_to_template_1Warp.nii.gz"'
    assert scan_source(src) == []


def test_ignores_prose():
    """Comments never reach the AST; docstrings are skipped explicitly."""
    src = (
        '"""We resolve tpl-to-SIGMA_1Warp.nii.gz rather than naming it."""\n'
        '# see tpl-to-SIGMA_0GenericAffine.mat\n'
        'def f():\n'
        '    """Also mentions tpl-to-SIGMA_1Warp.nii.gz."""\n'
        '    return 1\n'
    )
    assert scan_source(src) == []


def test_opt_out_comment_suppresses_one_line():
    src = (
        f'a = "tpl-to-SIGMA_1Warp.nii.gz"  {OPT_OUT}\n'
        'b = "tpl-to-SIGMA_0GenericAffine.mat"\n'
    )
    found = scan_source(src)
    assert [f.line for f in found] == [2]


def test_unparseable_file_is_skipped_not_raised():
    """A study repo will contain half-written scripts; the check must not die on one."""
    assert scan_source("def (((") == []


def test_scans_a_directory_tree(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "bad.py").write_text('x = "tpl-to-SIGMA_1Warp.nii.gz"')
    (tmp_path / "pkg" / "good.py").write_text("from neurofaune.templates.sigma_warp import resolve_tpl_to_sigma")
    (tmp_path / "pkg" / "notpython.txt").write_text('x = "tpl-to-SIGMA_1Warp.nii.gz"')

    found = find_hardcoded_sigma_paths([tmp_path])
    assert len(found) == 1
    assert found[0].file.name == "bad.py"


def test_neurofaune_itself_is_clean():
    """The package routes every caller through the resolver.

    Was 18 findings across 5 files when this check first ran on 2026-08-24 --
    all consumer-side, none of which resolved on a study carrying ANTs' real
    output names. All fixed; this now holds the line at zero.
    """
    root = Path(__file__).resolve().parents[2]
    found = find_hardcoded_sigma_paths([root / "neurofaune"])
    assert not found, (
        "hardcoded tpl->SIGMA path(s) — use resolve_tpl_to_sigma_for_cohort():\n"
        + "\n".join(str(f) for f in found)
    )
