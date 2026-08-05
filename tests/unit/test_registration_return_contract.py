"""Both registration branches must return the same keys.

register_bold_to_template has two paths -- anat composition (preferred) and a
direct NCC fallback -- and the caller writes every key into the registration
sidecar. When the composition branch omitted three of them, the caller raised
KeyError AFTER a successful registration: the composite warp was written and
used, but the sidecar kept a months-old timestamp, the registration QC never
ran, and the whole thing surfaced as a caught "Registration failed" that did
not fail the session.
"""
import ast
import inspect
from pathlib import Path

import pytest

import neurofaune.preprocess.workflows.func_preprocess as fp
import neurofaune.preprocess.workflows.msme_preprocess as mp


def _returned_key_sets(func):
    """Every dict-literal `return` in a function, as sets of its literal keys."""
    src = inspect.getsource(func)
    tree = ast.parse(inspect.cleandoc(src.split("\n", 1)[0] + "\n" + src.split("\n", 1)[1]))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            keys = {k.value for k in node.value.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)}
            if keys:
                out.append(keys)
    return out


# The sidecar the caller builds needs all of these.
SIDECAR_KEYS = {"affine_transform", "template_file", "bold_shape", "template_shape"}


def test_bold_registration_branches_agree():
    key_sets = _returned_key_sets(fp.register_bold_to_template)
    assert len(key_sets) >= 2, "expected a composition branch and a fallback"
    for keys in key_sets:
        missing = SIDECAR_KEYS - keys
        assert not missing, f"a return branch omits {sorted(missing)}"


def test_every_branch_satisfies_what_the_caller_subscripts():
    """Derive the contract from the call site, not from a hand-kept list.

    Branches may return extra keys (the composition path adds `method` and
    `moving_to_anat_affine`); what they may not do is omit one the caller
    accesses with [] -- that is the KeyError. Optional .get() access is fine.
    """
    source = Path(inspect.getfile(fp)).read_text()
    tree = ast.parse(source)

    def _calls_bold_reg(node):
        return any(
            isinstance(n, ast.Call) and getattr(n.func, "id", None) ==
            "register_bold_to_template"
            for n in ast.walk(node)
        )

    # Scope to the try-block that calls register_bold_to_template. The module
    # reuses the name `registration_results` for the separate BOLD->T2w
    # registration, whose keys (t2w_file, ...) are a different contract.
    required = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Try) and _calls_bold_reg(node)):
            continue
        for sub in ast.walk(node):
            if (isinstance(sub, ast.Subscript)
                    and isinstance(sub.value, ast.Name)
                    and sub.value.id == "registration_results"
                    and isinstance(sub.slice, ast.Constant)
                    and isinstance(sub.slice.value, str)):
                required.add(sub.slice.value)

    assert required, "found no registration_results[...] accesses to check"

    for keys in _returned_key_sets(fp.register_bold_to_template):
        missing = required - keys
        assert not missing, (
            f"caller subscripts {sorted(missing)} that this branch never returns"
        )


def test_msme_registration_branches_agree():
    """MSME had this right; keep it that way."""
    msme_keys = {"affine_transform", "template_file", "msme_shape", "template_shape"}
    for keys in _returned_key_sets(mp.register_msme_to_template):
        missing = msme_keys - keys
        assert not missing, f"a return branch omits {sorted(missing)}"
