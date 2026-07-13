"""Pin the slice-distance z-spacing fix in Bruker parsing.

The NIfTI z voxel spacing must be the center-to-center slice DISTANCE
(thickness + gap), not the thickness alone — otherwise gapped
(non_contiguous) acquisitions are z-compressed and spatial registration
breaks (observed: func 0.6 vs true 0.8, MSME 0.8 vs true 1.0).
"""
from neurofaune.utils.fix_bruker_voxel_sizes import parse_bruker_method

_GAPPED = """##$PVM_SpatResol=( 2 )
0.4 0.4
##$PVM_SliceThick=0.6
##$PVM_SPackArrSliceGap=( 1 )
0.2
##$PVM_SPackArrSliceDistance=( 1 )
0.8
"""

_CONTIGUOUS = """##$PVM_SpatResol=( 2 )
0.5 0.5
##$PVM_SliceThick=0.5
"""


def test_z_spacing_uses_slice_distance(tmp_path):
    m = tmp_path / "method"
    m.write_text(_GAPPED)
    p = parse_bruker_method(m)
    assert p["slice_thickness"] == 0.6
    assert p["slice_distance"] == 0.8
    assert p["voxel_size"][2] == 0.8   # distance, NOT thickness


def test_z_spacing_falls_back_to_thickness_when_no_distance(tmp_path):
    m = tmp_path / "method"
    m.write_text(_CONTIGUOUS)
    p = parse_bruker_method(m)
    assert "slice_distance" not in p
    assert p["voxel_size"][2] == 0.5   # thickness (no gap reported)
