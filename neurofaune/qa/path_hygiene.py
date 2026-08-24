"""Fail when code names a template->SIGMA transform file instead of resolving it.

WHY THIS EXISTS
---------------
The tpl->SIGMA transform pair has no single canonical filename. ``antsRegistration``
names its output after whatever ``--output`` prefix the caller passed, so a study
that registers per-timepoint templates ends up with, say::

    templates/p60/tpl-CPZp60_to-SIGMA_0GenericAffine.mat

while another study -- or the same study before someone changed the prefix --
has::

    templates/p60/transforms/tpl-to-SIGMA_0GenericAffine.mat

``neurofaune.templates.sigma_warp.resolve_tpl_to_sigma`` exists precisely to
absorb that variation: it searches the plausible directories and matches the
affine by *suffix*, then pairs the warp to it by prefix. Every caller that goes
through the resolver is immune to the layout changing under it.

Code that hardcodes the filename instead is not immune, and it fails in the worst
available way. A missing transform is not a crash -- the chain is assembled from
paths that simply do not exist, the step reports that it "skipped" or returns
None, and the run continues to completion. On the cuprizone study this cost two
separate rebuilds:

  * 2026-08-20 -- the DKI/NODDI maps never reached SIGMA space, because the
    hardcoded lookup ran before the layout it assumed was written. 4 of 11 maps
    emitted, exit code 0.
  * 2026-08-22 -- ``qc_anat_registration.py`` kept its own copy of the path
    logic. A rebuild replaced the hand-made ``transforms/`` symlink farm with
    ANTs' real output names; all 92 sessions then failed the T2w->SIGMA chain
    while the dwi, func and msme arms sailed through on the resolver.

Both were found by reading a log, not by anything failing. Hence a check.

USE
---
    from neurofaune.qa import find_hardcoded_sigma_paths
    findings = find_hardcoded_sigma_paths([Path("preprocessing/code")])

or from a shell, over a study's own code::

    neurofaune check-paths preprocessing/code

Exits non-zero when anything is found, so it can gate a pipeline driver.

OPTING OUT
----------
A line may end with ``# sigma-path: ok`` when it genuinely must name the file --
``sigma_warp`` itself does, since it is the module defining the convention.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

#: Matches a tpl->SIGMA transform filename in any of its spellings: with or
#: without a study prefix (``tpl-CPZp60_to-SIGMA_...``), and for either member of
#: the pair. Deliberately NOT anchored, so it still fires inside an f-string
#: fragment such as f"tpl-{cohort}_to-SIGMA_1Warp.nii.gz".
#:
#: Kept narrow on purpose: a bare ``_0GenericAffine.mat`` is a perfectly normal
#: thing to name, because per-session modality->template affines are not subject
#: to the layout ambiguity this check is about. Only the SIGMA leg is.
SIGMA_TRANSFORM_RE = re.compile(r"to-SIGMA_\d*(?:GenericAffine|InverseWarp|Warp)")

#: Line-level escape hatch.
OPT_OUT = "# sigma-path: ok"

#: Files allowed to name the transforms outright, matched by path suffix.
#: ``sigma_warp`` defines the constants; the checker names them in its own regex
#: and docstring; the tests assert on them.
DEFAULT_ALLOW: tuple[str, ...] = (
    "neurofaune/templates/sigma_warp.py",
    "neurofaune/qa/path_hygiene.py",
)


@dataclass(frozen=True)
class Finding:
    """One hardcoded SIGMA transform path."""

    file: Path
    line: int
    literal: str

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return (
            f"{self.file}:{self.line}: hardcoded tpl->SIGMA transform "
            f"{self.literal!r}\n"
            f"    use neurofaune.templates.sigma_warp.resolve_tpl_to_sigma() instead; "
            f"ANTs names these after the --output prefix, so this path is not stable.\n"
            f"    if this line genuinely must name the file, end it with '{OPT_OUT}'."
        )


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """id()s of the string Constants that are docstrings, so prose is not flagged.

    Comments never reach the AST at all, so they need no handling here.
    """
    out: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                and isinstance(first.value.value, str):
            out.add(id(first.value))
    return out


def scan_source(text: str, filename: str | Path = "<string>") -> List[Finding]:
    """Find hardcoded tpl->SIGMA transform names in one Python source string.

    Works on the AST rather than by grepping, so a mention in a comment or a
    docstring is not a finding -- only a string the code actually evaluates.
    """
    path = Path(filename)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []

    lines = text.splitlines()
    skip = _docstring_nodes(tree)
    findings: List[Finding] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or id(node) in skip:
            continue
        if not isinstance(node.value, str):
            continue
        if not SIGMA_TRANSFORM_RE.search(node.value):
            continue
        lineno = getattr(node, "lineno", 0)
        source_line = lines[lineno - 1] if 0 < lineno <= len(lines) else ""
        if OPT_OUT in source_line:
            continue
        findings.append(Finding(file=path, line=lineno, literal=node.value))

    return findings


def find_hardcoded_sigma_paths(
    roots: Iterable[Path],
    allow: Sequence[str] = DEFAULT_ALLOW,
) -> List[Finding]:
    """Scan .py files under ``roots``. A root may also be a single file."""
    findings: List[Finding] = []
    for root in roots:
        root = Path(root)
        files = [root] if root.is_file() else sorted(root.rglob("*.py"))
        for f in files:
            posix = f.as_posix()
            if any(posix.endswith(a) for a in allow):
                continue
            try:
                text = f.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            findings.extend(scan_source(text, f))
    return findings
