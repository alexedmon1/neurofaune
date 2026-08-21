#!/usr/bin/env python
"""Export an fsl_mrs fit as plain CSV curves, for figure-making.

Run under FSL's own Python interpreter, like ``_fsl_preproc.py``.

Why this exists
---------------
The ``fsl_mrs`` command line writes ``report.html`` (interactive plotly) and
``fit_summary.png``, but no fit curves as data. Building a custom or
group-level figure from that means scraping the HTML or re-running the fit by
hand. LCModel, by contrast, writes its curves to a ``.coord`` text file.

This closes that gap for the FSL-MRS path: it repeats the fit through the
Python API with the same settings the command line used -- deterministic, so
the curves correspond to the reported concentrations -- and writes:

    {prefix}_fit-curves.csv       ppm, data, fit, baseline, residual
    {prefix}_fit-metabolites.csv  ppm plus one column per basis metabolite

Both are real-valued spectra over the fit range, ready to plot directly.
"""

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, help='Preprocessed NIfTI-MRS')
    parser.add_argument('--basis', required=True, help='FSL-MRS basis directory')
    parser.add_argument('--output-prefix', required=True,
                        help='Path stem; _fit-curves.csv etc. are appended')
    parser.add_argument('--ppmlim', nargs=2, type=float, default=[0.2, 4.2])
    parser.add_argument('--baseline', default='poly,4')
    parser.add_argument('--metab-groups', nargs='*', default=['NAA'])
    parser.add_argument('--free-shift', action='store_true')
    args = parser.parse_args()

    from pathlib import Path

    import numpy as np
    import pandas as pd
    from fsl_mrs.utils import fitting, misc, mrs_io

    data = mrs_io.read_FID(args.data)
    mrs = data.mrs(basis_file=args.basis)
    mrs.processForFitting()

    ppmlim = (min(args.ppmlim), max(args.ppmlim))
    fit_args = {
        'ppmlim': ppmlim,
        'method': 'Newton',
        'metab_groups': misc.parse_metab_groups(mrs, args.metab_groups),
        'baseline': args.baseline,
        'model': 'free_shift' if args.free_shift else 'voigt',
    }
    res = fitting.fit_FSLModel(mrs, **fit_args)

    ppm = mrs.getAxes(ppmlim=ppmlim)
    first, last = misc.limit_to_range(mrs.getAxes(), ppmlim)

    def real_spec(spectrum):
        return np.real(np.asarray(spectrum)[first:last])

    # res.baseline and res.residuals are FIDs, not spectra -- only pred_spec is
    # already transformed (`_baseline = predictedFID(..., mode='Baseline')`,
    # `_residuals = mrs.FID - pred`). Taking the real part of a time-domain
    # array and labelling it a spectrum yields noise-shaped nonsense, so they
    # are transformed here before slicing.
    curves = pd.DataFrame({
        'ppm': ppm,
        'data': real_spec(mrs.get_spec()),
        'fit': real_spec(res.pred_spec),
        'baseline': real_spec(misc.FIDToSpec(res.baseline)),
        'residual': real_spec(misc.FIDToSpec(res.residuals)),
    })
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    curves.to_csv(f'{prefix}_fit-curves.csv', index=False)

    # Per-metabolite contributions, so individual species can be shown under
    # the total fit the way LCModel's plots do.
    # Iterate mrs.names, not res.metabs: the latter includes combined species
    # (Cr+PCr and friends) that have no basis spectrum of their own.
    per_metab = {'ppm': ppm}
    for name in mrs.names:
        fid = res.predictedFID(mrs, mode=name, noBaseline=True)
        per_metab[name] = real_spec(misc.FIDToSpec(fid))
    pd.DataFrame(per_metab).to_csv(f'{prefix}_fit-metabolites.csv', index=False)

    print(f'... wrote {prefix}_fit-curves.csv and {prefix}_fit-metabolites.csv')
    return 0


if __name__ == '__main__':
    sys.exit(main())
