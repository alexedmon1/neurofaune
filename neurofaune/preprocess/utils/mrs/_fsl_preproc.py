#!/usr/bin/env python
"""SVS preprocessing chain, run under FSL's own Python interpreter.

This is deliberately not importable from neurofaune's environment: ``fsl_mrs``
lives inside the FSL installation, so this module is executed as a subprocess
by :func:`neurofaune.preprocess.workflows.mrs_preprocess.run_internal_preproc`
via ``$FSLDIR/bin/python``. That keeps the "no separate conda environment"
property while still giving step-level control.

Why not just call ``fsl_mrs_preproc``
-------------------------------------
It runs the same chain and then finishes with:

    shift_to_reference(data, 3.027, (2.9, 3.1))
    phase_correct(data, (2.9, 3.1))

Both take ``argmax(|spectrum|)`` inside that hardcoded window, move that point
to 3.027 and phase it to zero. When the wrong point wins -- which happened on
6-7 of 53 cuprizone sessions -- the whole spectrum is displaced in ppm and
given an arbitrary global phase, and no reference metabolite can be fit
afterwards. The window is not adjustable from the command line.

This chain drops ``shift_to_reference`` entirely: the converter has already
referenced the spectrum on tCr, and more robustly, over a wide search window
cross-checked against NAA rather than a 0.2 ppm window with no validation (see
``bruker_mrs.measure_metabolite_offset``).

Phasing is kept, because leaving it to ``fsl_mrs`` as a free parameter costs
about 30% of the fitted SNR. ``--phase-method`` selects how:

``search`` (default)
    Scan zero-order phase over the full circle, scoring the whole metabolite
    band for absorptive character. See :func:`search_phase` for why the band
    and the full circle both matter.

``tcr``
    ``phase_correct`` on the creatine peak alone, over 2.95-3.10 ppm. Safe here
    for the reason the stock version isn't: with tCr already at 3.027 +/- 0.001
    the peak it lands on is the one intended, rather than whatever is tallest
    in a window the spectrum may have drifted out of.

``--no-phase`` skips the step entirely and leaves the phase to ``fsl_mrs``.
"""

import argparse
import sys


def search_phase(data, ppmlim=(0.5, 4.2), penalty=2.0, coarse_step=1.0):
    """Zero-order phase by searching for the most absorptive spectrum.

    Phasing on a single peak uses one point of a noisy spectrum and inherits
    that point's noise. This instead scores the whole metabolite region: a
    correctly phased spectrum is predominantly positive there, so the score
    rewards positive real signal and penalises the negative lobes that a wrong
    phase produces.

    The search covers the full circle, which is the point of it. fsl_mrs fits
    phase by local descent from zero with concentrations bounded non-negative,
    so a spectrum near 180 degrees out cannot be recovered by the fit -- the
    metabolites simply go to zero. LCModel avoids that trap differently, by
    learning how wide to make its phase prior rather than assuming the
    correction is small (its manual singles out Bruker data as a case where
    "the zero-order phase correction is often not small"). Since fsl_mrs'
    optimiser cannot be changed, the equivalent is done here instead.
    """
    import numpy as np
    from fsl_mrs.utils.preproc import nifti_mrs_proc as proc

    fid = np.asanyarray(data[:]).squeeze()
    spectrum = np.fft.fftshift(np.fft.fft(fid))
    frequency = np.fft.fftshift(np.fft.fftfreq(spectrum.size, data.dwelltime))
    # Same ppm convention as the converter, verified against Bruker's own
    # reconstruction (NAA at 2.01, tCr at 3.03).
    ppm = frequency / data.spectrometer_frequency[0] + 4.65
    band = (ppm > min(ppmlim)) & (ppm < max(ppmlim))
    in_band = spectrum[band]

    def score(angle_deg):
        rotated = (in_band * np.exp(1j * np.deg2rad(angle_deg))).real
        return rotated.sum() - penalty * np.abs(np.minimum(rotated, 0.0)).sum()

    coarse = np.arange(-180.0, 180.0, coarse_step)
    best = float(coarse[int(np.argmax([score(a) for a in coarse]))])
    fine = np.arange(best - coarse_step, best + coarse_step, 0.05)
    best = float(fine[int(np.argmax([score(a) for a in fine]))])

    print(f'... zero-order phase search: {best:+.2f} deg')
    return proc.apply_fixed_phase(data, best)


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
    parser.add_argument('--no-phase', action='store_true',
                        help='Skip zero-order phasing and leave it to fsl_mrs')
    parser.add_argument('--phase-method', choices=('search', 'tcr'), default='search',
                        help="'search' scans zero-order phase for the most "
                             "absorptive spectrum; 'tcr' phases on the creatine "
                             "peak alone")
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

    if not args.no_phase:
        if args.phase_method == 'tcr':
            # Phase on the creatine peak alone. Safe here only because the
            # converter has already put tCr at 3.027 +/- 0.001.
            supp = proc.phase_correct(supp, (2.95, 3.10))
        else:
            supp = search_phase(supp)

    # Still no shift_to_reference -- see the module docstring.
    supp.save(str(output / 'metab.nii.gz'))
    ref.save(str(output / 'wref.nii.gz'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
