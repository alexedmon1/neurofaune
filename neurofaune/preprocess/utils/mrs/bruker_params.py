"""Generic JCAMP-DX reader for Bruker ``method`` / ``acqp`` parameter files.

The existing :func:`neurofaune.utils.fix_bruker_voxel_sizes.parse_bruker_method`
pulls a fixed set of keys with per-key regexes. Spectroscopy needs many more
parameters (voxel geometry, spectral width, water-suppression state, the
embedded reference FID), so this module parses the whole file once into a
dict instead.

Values are returned as:

- ``int`` / ``float`` for numeric scalars
- ``str`` for symbolic enums (``VAPOR``) and ``<bracketed>`` strings
- :class:`numpy.ndarray` for numeric arrays, reshaped to the declared shape
- ``list`` of ``str`` for arrays that contain non-numeric tokens
- ``str`` (raw text) for structured values PARX writes as nested tuples
  (e.g. ``PVM_SliceGeo``), which have no useful generic representation
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

# ``##$KEY=`` at the start of a line, everything after ``=`` is the value head.
_KEY_RE = re.compile(r'^##\$([A-Za-z0-9_]+)=(.*)$')
# ``( 8 )`` or ``( 1, 3, 3 )`` -- the declared shape of an array value.
_SHAPE_RE = re.compile(r'^\(\s*(\d+(?:\s*,\s*\d+)*)\s*\)\s*$')
# PARX run-length encoding for constant arrays: ``@41*(1)``.
_RLE_RE = re.compile(r'^@(\d+)\*\(([^)]*)\)$')


def _to_number(token: str) -> Optional[Union[int, float]]:
    """Return ``token`` as an int/float, or None if it isn't numeric."""
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        return None


def _expand_rle(tokens: List[str]) -> List[str]:
    """Expand PARX run-length tokens: ``@41*(1)`` becomes 41 copies of ``1``."""
    expanded: List[str] = []
    for token in tokens:
        match = _RLE_RE.match(token)
        if match is None:
            expanded.append(token)
        else:
            expanded.extend([match.group(2)] * int(match.group(1)))
    return expanded


def _parse_value(head: str, body_lines: List[str]) -> Any:
    """Turn a raw ``##$KEY=`` head plus its continuation lines into a value."""
    head = head.strip()
    shape_match = _SHAPE_RE.match(head)

    if shape_match is None:
        # Scalar on the same line: number, enum symbol, or <string>.
        if head.startswith('<') and head.endswith('>'):
            return head[1:-1]
        number = _to_number(head)
        return head if number is None else number

    shape = tuple(int(n) for n in shape_match.group(1).split(','))
    body = '\n'.join(body_lines).strip()

    # A single ``<...>`` payload is a string, not an array of characters --
    # the declared size is just the buffer length PARX reserved.
    if body.startswith('<') and body.endswith('>') and body.count('<') == 1:
        return body[1:-1]

    # Structured/nested values (PVM_SliceGeo, PVM_VoxelGeoCub, ACQ_jobs ...).
    # These are tuples-of-tuples with no generic shape; hand back the raw text.
    if body.startswith('('):
        return body

    tokens = _expand_rle(body.split())
    numbers = [_to_number(t) for t in tokens]
    if any(n is None for n in numbers):
        return tokens

    array = np.array(numbers, dtype=float)
    if int(np.prod(shape)) == array.size and len(shape) > 1:
        array = array.reshape(shape)
    return array


def read_jcampdx(path: Path) -> Dict[str, Any]:
    """Parse a Bruker JCAMP-DX parameter file into a dict.

    Parameters
    ----------
    path : Path
        Path to a Bruker ``method``, ``acqp`` or ``visu_pars`` file.

    Returns
    -------
    dict
        Mapping of parameter name (without the ``##$`` prefix) to value.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Bruker parameter file not found: {path}")

    params: Dict[str, Any] = {}
    key: Optional[str] = None
    head = ''
    body: List[str] = []

    with open(path, 'r', errors='replace') as handle:
        for line in handle:
            line = line.rstrip('\n')
            match = _KEY_RE.match(line)
            if match is not None:
                if key is not None:
                    params[key] = _parse_value(head, body)
                key, head, body = match.group(1), match.group(2), []
                continue
            # ``$$`` are comments and ``##`` non-``$`` records (##END= etc.)
            # both terminate the value currently being accumulated.
            if line.startswith('$$'):
                continue
            if line.startswith('##'):
                if key is not None:
                    params[key] = _parse_value(head, body)
                    key, head, body = None, '', []
                continue
            if key is not None:
                body.append(line)

    if key is not None:
        params[key] = _parse_value(head, body)

    return params


def read_scan_params(scan_dir: Path) -> Dict[str, Any]:
    """Read a scan's ``method`` and ``acqp``, merged into one dict.

    ``method`` wins on key collisions -- it holds the PVM-level parameters that
    describe the sequence as the user set it up, while ``acqp`` holds the
    lower-level acquisition view of the same quantities.
    """
    scan_dir = Path(scan_dir)
    params: Dict[str, Any] = {}
    acqp = scan_dir / 'acqp'
    if acqp.exists():
        params.update(read_jcampdx(acqp))
    params.update(read_jcampdx(scan_dir / 'method'))
    return params
