"""Gate-tier regression: preprocessing *workflow assembly* (tool-free).

This demonstrates how to gate a preprocessing refactor without running ANTs/
FSL: freeze the assembled nipype graph (nodes, interfaces, wiring) as a golden
and assert a candidate still emits it. The demo builds a tiny tool-free
workflow; for real coverage, replace ``_demo_workflow`` with a call into a
neurofaune workflow builder (e.g. neurofaune.preprocess.workflows.anat_preprocess)
and name the golden after it.
"""

from __future__ import annotations

import pytest

from tests.regression._workflow import assert_workflow_matches_golden

pytestmark = pytest.mark.regression


def _scale(in_val, factor):
    return in_val * factor


def _demo_workflow():
    import nipype.pipeline.engine as pe
    from nipype.interfaces.utility import Function, IdentityInterface

    inputnode = pe.Node(IdentityInterface(fields=["x"]), name="inputnode")
    scale = pe.Node(
        Function(input_names=["in_val", "factor"], output_names=["out_val"], function=_scale),
        name="scale",
    )
    outputnode = pe.Node(IdentityInterface(fields=["y"]), name="outputnode")

    wf = pe.Workflow(name="demo")
    wf.connect([
        (inputnode, scale, [("x", "in_val")]),
        (scale, outputnode, [("out_val", "y")]),
    ])
    return wf


def test_demo_workflow_graph_frozen():
    assert_workflow_matches_golden(_demo_workflow(), "preprocess_demo")
