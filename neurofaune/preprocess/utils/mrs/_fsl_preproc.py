#!/usr/bin/env python
"""SVS preprocessing chain, run under FSL's own Python interpreter.

This is deliberately not importable from neurofaune's environment: ``fsl_mrs``
lives inside the FSL installation, so this module is executed as a subprocess
by :func:`neurofaune.preprocess.workflows.mrs_preprocess.run_internal_preproc`
via ``$FSLDIR/bin/python``. That keeps the "no separate conda environment"
property while still giving step-level control.

Why not just call ``fsl_mrs_preproc``
-------------------------------------
It runs the same chain and then finishes with two steps this one omits:

    shift_to_reference(data, 3.027, (2.9, 3.1))
    phase_correct(data, (2.9, 3.1))

Both take ``argmax(|spectrum|)`` inside that hardcoded window, move that point
to 3.027 and phase it to zero. When the wrong point wins -- which happened on
6-7 of 53 cuprizone sessions -- the whole spectrum is displaced in ppm and
given an arbitrary global phase, and no reference metabolite can be fit
afterwards. The window is not adjustable from the command line.

Skipping both is safe here because the converter already references the
spectrum on tCr, more robustly: a wide search window cross-checked against
NAA, rather than a 0.2 ppm window with no validation (see
``bruker_mrs.measure_metabolite_offset``). Zero-order phase is left to
``fsl_mrs``, which fits it as a free parameter. On the three sessions tested
that ``fsl_mrs_preproc`` could not produce a fittable spectrum for, this chain
yields clean upright spectra with NAA SNR 13.0-13.2.
"""

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, help='Water-suppressed NIfTI-MRS')
    parser.add_argument('--reference', required=True, help='Water reference NIfTI-MRS')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--align-window', type=int, default=32,
                        help='Align dynamics in windows of this many shots; 0 disables')
    parser.add_argument('--no-removal', action='store_true',
                        help='Keep averages unlike the rest')
    parser.add_argument('--remove-water', action='store_true',
                        help='HLSVD residual water removal')
    args = parser.parse_args()

    from pathlib import Path

    from fsl_mrs.utils import mrs_io
    from fsl_mrs.utils.preproc import nifti_mrs_proc as proc

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    supp = mrs_io.read_FID(args.data)
    ref = mrs_io.read_FID(args.reference)

    # Coil combination, using the water reference for the per-channel phase
    # and weighting. Both arrays must be combined the same way for the later
    # eddy-current correction to mean anything.
    supp = proc.coilcombine(supp, reference=ref)
    ref = proc.coilcombine(ref, reference=ref)

    # Align the individual shots to each other. Single rodent averages are too
    # noisy to align one at a time, hence the window.
    if args.align_window > 0:
        supp = proc.align(supp, 'DIM_DYN', ppmlim=(0.2, 4.2),
                          window=args.align_window)

    if not args.no_removal:
        supp = proc.remove_unlike(supp)[0]

    supp = proc.average(supp, 'DIM_DYN')
    ref = proc.average(ref, 'DIM_DYN')

    # Eddy-current correction. Applying it to the reference against itself
    # also removes the reference's own phase, which is what the water peak
    # would otherwise need a separate phasing step for.
    supp = proc.ecc(supp, ref)
    ref = proc.ecc(ref, ref)

    if args.remove_water:
        supp = proc.remove_peaks(supp, [-0.25, 0.25], limit_units='ppm')

    # No shift_to_reference or phase_correct here -- see the module docstring.
    supp.save(str(output / 'metab.nii.gz'))
    ref.save(str(output / 'wref.nii.gz'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
