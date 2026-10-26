"""Susceptibility / implant artifact QC.

Measures how much an implant on the skull (EEG electrode array, headpost,
cement) degrades each MRI modality, as a function of depth below the dorsal
brain surface:

- ``maps``    -- pure-numpy metrics: dorsal depth, depth profiles, tSNR,
                 normalised signal ratios.
- ``topup``   -- run FSL topup on a reverse-phase-encode pair using the BIDS
                 readout fields neurofaune writes for Bruker EPI.
- ``implant`` -- the per-session driver: B0, distortion, dropout, tSNR and
                 chemical-shift checks against a distortion-free T2w reference.
"""
