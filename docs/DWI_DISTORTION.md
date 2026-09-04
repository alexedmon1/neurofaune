# EPI Distortion in Rodent DWI: When It Matters and How to Check

Echo-planar imaging distorts along the phase-encode (PE) axis. For scalar ROI
means the effect is usually tolerable; for tractography and for any analysis
that depends on where a boundary sits (TBSS skeletons, connectome node
assignment) it displaces the thing being measured.

This guide covers two questions worth answering before committing to a
registration strategy: **can you correct distortion**, and **do you need to**.

---

## 1. Can you correct it? Check for a true reverse-PE acquisition

`topup` requires two acquisitions of the same volume with **opposite blip
polarity**, both phase-encoded. A second b0 with matching geometry is *not*
sufficient — it is often a reference scan acquired with phase encoding
disabled, which carries no distortion information to difference against.

On Bruker, read the `method` file of both scans and compare:

```bash
diff <(grep -v 'PVM_Dw' <scan_A>/method) <(grep -v 'PVM_Dw' <scan_B>/method) \
  | grep -iE 'blip|phase|epi'
```

The decisive field is `PVM_EpiBlipsOff`:

| `PVM_EpiBlipsOff` | meaning |
| --- | --- |
| `No` | phase encoding active — a real acquisition |
| `Yes` | **no phase encoding** — a reference scan, unusable for topup |

A reverse-PE pair shows blips *on* in both, differing in polarity or in
`PVM_SPackArrPhase1Offset` / gradient sign — not one on and one off.

> **Worked example (cuprizone, 2026-09).** The DWI (`run-10`) has
> `PVM_EpiBlipsOff=No`; its companion single-b0 scan (`run-18`) has
> `PVM_EpiBlipsOff=Yes`. Same geometry, same `PVM_SPackArrReadOrient=L_R`, but
> the companion is a blips-off reference. **topup is not available for this
> study.** The two also differ in `PVM_EpiNShots` (4 vs 2), confirming they are
> different acquisitions rather than a PE-reversed pair.

---

## 2. Do you need to? The PE-versus-readout asymmetry test

If you cannot correct distortion, the next question is how much there is. The
useful property: **distortion acts only along the phase-encode axis.** Ordinary
registration error and skull-stripping differences are roughly isotropic, so
comparing mismatch along PE against mismatch along readout separates them.

Determine the PE axis from `PVM_SPackArrReadOrient` — PE is the *other*
in-plane axis (readout `L_R` means PE is A–P).

Then, for each slice, compare the DWI brain mask against the anatomical brain
mask warped into DWI space, measuring extent separately along each in-plane
axis:

```
ratio = p95(|PE-axis extent mismatch|) / p95(|readout-axis extent mismatch|)
```

- **ratio > 1** — mismatch is PE-specific. Distortion is real and worth
  correcting or modelling.
- **ratio ≤ 1** — mismatch is not distortion-shaped. It is dominated by
  skull-stripping differences between modalities and by registration residual,
  neither of which distortion correction addresses.

Two cautions. Use **percentile-based extents (2nd–98th), never a bounding
box**: min/max is destroyed by a single stray mask voxel, which on this study
produced an implausible 8.5 mm median before the measure was made robust. And
this is a proxy — it cannot fully separate distortion from mask-definition
differences. The definitive measurement is a SyN b0→T2w registration, reading
the PE-axis component of the resulting warp field.

> **Worked example (cuprizone).** Across six sessions the ratio was 0.20, 0.27,
> 0.32, 0.37, 0.39, 0.40 — consistently **below 1**, so mismatch is
> concentrated in the readout direction, which distortion cannot cause.
> Corroborated mechanically: the sequence is 4-shot **segmented** EPI
> (`PVM_EpiNShots=4`, `DTI_EPI_seg`), which shortens effective echo spacing
> roughly fourfold relative to single-shot and shrinks distortion in
> proportion. Conclusion: distortion is not the dominant source of DWI-to-
> anatomy mismatch in this study.

Segmented (multi-shot) EPI is common in rodent protocols and is the main reason
rodent DWI often distorts far less than the single-shot human case would lead
you to expect. Check `PVM_EpiNShots` before assuming otherwise.

---

## 3. What this implies for registration

If distortion is small, an affine DWI→template transform is defensible and a
nonlinear one buys mostly better boundary agreement rather than distortion
correction. If distortion is large and uncorrectable, a nonlinear registration
partly absorbs it — which is why
`run_dwi_preprocessing` propagates the brain mask from the same-session T2w by
SyN specifically (see the "Post-eddy brain mask refinement" step).

Note the contrast trap when going nonlinear: `register_fa_to_template` registers
**FA** to a **T2w** template, and those have an inverted intensity relationship
(white matter is bright in FA, dark in T2w). `antsRegistrationSyN.sh` uses
cross-correlation for its SyN stage, which assumes matching contrast, so a
naive nonlinear upgrade can warp white matter badly while reporting success.
Prefer one of:

1. **Build a study FA template** and register FA→FA-template within-modality,
   then register that template to SIGMA once. This removes the contrast problem
   rather than compensating for it, and optimises white-matter correspondence
   directly — which is what TBSS skeletonisation and connectome node placement
   both depend on:

   ```python
   from neurofaune.templates.builder import build_dwi_template

   build_dwi_template(
       derivatives_dir=Path("/study/derivatives"),
       output_dir=Path("/study/templates/dwi/p60"),
       cohort="p60", study_code="CPZ",
       exclude=qc_exclusions,   # a blurred template degrades every subject
       max_subjects=35,
   )
   ```

   With an FA template as the target, `metric="CC"` becomes valid and is
   sharper than MI.

2. **Register the b0 rather than FA.** The b0 is T2-weighted, so it matches
   T2w-template contrast directly. Cheap, and needs no new template:

   ```python
   register_fa_to_template(..., moving_file=mean_b0, transform_type="s")
   ```

3. **Use mutual information** if registering FA to a T2w template
   nonlinearly is unavoidable. This is the default (`metric="MI"`), and
   neurofaune drops to a direct `antsRegistration` call to honour it, because
   `antsRegistrationSyN.sh` cannot.

Enable the deformable path per study rather than in code:

```yaml
diffusion:
  registration:
    transform_type: "a"   # 's' for rigid + affine + SyN
    metric: "MI"          # 'CC' only when the target is an FA template
```

The default remains affine, so existing studies are unaffected until they opt in.

See `docs/ATLAS_GUIDE.md` for the `modality → cohort template → SIGMA` chain
these transforms feed.
