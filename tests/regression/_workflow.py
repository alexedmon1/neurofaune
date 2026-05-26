"""Freeze a nipype workflow's *structure* as a golden — without running tools.

The boundary of a preprocessing refactor is often not the image output (which
isn't bit-reproducible across ANTs/FSL versions) but the workflow it assembles:
which nodes, with which parameters, wired in which order, and the command-line
each node would emit. That is fully deterministic and tool-free, so it belongs
in the fast blocking gate.

Usage in a @pytest.mark.regression test:

    from tests.regression._workflow import assert_workflow_matches_golden
    wf = build_anat_workflow(config)         # candidate's workflow builder
    assert_workflow_matches_golden(wf, "anat_preprocess")

Regenerate deliberately with NEUROFAUNE_UPDATE_GOLDEN=1 (see _equivalence).
"""

from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).parent
GOLDEN = HERE / "golden"
UPDATE = os.environ.get("NEUROFAUNE_UPDATE_GOLDEN") == "1"


def describe_workflow(wf) -> str:
    """Deterministic, diff-able text description of a nipype workflow graph."""
    lines = ["nodes:"]
    for name in sorted(wf.list_node_names()):
        node = wf.get_node(name)
        iface = getattr(node, "interface", None)
        kind = type(iface).__name__ if iface is not None else type(node).__name__
        lines.append(f"  {name} :: {kind}")

    edges: list[str] = []
    for u, v, data in wf._graph.edges(data=True):
        for src_out, dst_in in data.get("connect", []):
            edges.append(f"  {u.name}.{src_out} -> {v.name}.{dst_in}")
    lines.append("connections:")
    lines.extend(sorted(edges))
    return "\n".join(lines) + "\n"


def assert_workflow_matches_golden(wf, name: str) -> None:
    """Assert the workflow graph matches the committed golden description."""
    path = GOLDEN / f"{name}.workflow.txt"
    actual = describe_workflow(wf)
    if UPDATE or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(actual)
        if not UPDATE:
            raise AssertionError(
                f"workflow golden '{name}' did not exist and was created at {path}; "
                "review and commit it, then re-run."
            )
        return
    expected = path.read_text()
    assert actual == expected, (
        f"workflow graph for '{name}' changed.\n--- golden ---\n{expected}\n"
        f"--- candidate ---\n{actual}"
    )
