"""Mask helpers shared by the functional workflow, ICA classification and aCompCor."""

import numpy as np
from scipy import ndimage


def erosion_structure(inplane: bool = True) -> np.ndarray:
    """6-connected structuring element; ``inplane`` drops the through-slice neighbours.

    Slices are array axis 2 (the acquisition slices). A 3D erosion of a slab of
    thick slices removes whole slab-end slices -- scipy pads the border with 0, so
    every voxel on the first and last slice is "edge" -- and every voxel whose
    neighbour one slice (1 mm in rodent EPI) away is outside the brain. On the
    FAC600 mouse pilot one 3D iteration kept 71% (axial, 13 slices) and 44%
    (coronal, 9 slices) of the brain and emptied the end slices. In-plane erosion
    strips the in-plane rim only, and is the default for rat and mouse alike
    (``functional.erode_inplane``, default true); pass ``inplane=False`` for 3D.
    """
    s = ndimage.generate_binary_structure(3, 1)
    if inplane:
        s[:, :, 0] = False
        s[:, :, 2] = False
    return s


def erode(mask: np.ndarray, iterations: int, inplane: bool = True) -> np.ndarray:
    """Binary erosion with :func:`erosion_structure`; ``iterations <= 0`` returns the mask unchanged."""
    mask = np.asarray(mask, bool)
    if iterations <= 0:
        return mask
    return ndimage.binary_erosion(mask, structure=erosion_structure(inplane), iterations=iterations)
