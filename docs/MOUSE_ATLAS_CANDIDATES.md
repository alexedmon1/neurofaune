# Mouse atlases: in use, and candidates for neurofaune and source-localization

Compiled 2026-10-20 while building the mouse hub atlas (branch `feat/mouse-support`). The hub is
Antwerp space with CCFv3 labels via a Perens T2w bridge, plus the RABIES functional layer
(`/mnt/arborea/atlases/Antwerp/README.md`).

- **"Status" is as of that date.** Links were checked then, but hosts move: the DSURQE host was
  already gone.
- **BrainGlobe** packages (`https://gin.g-node.org/brainglobe/atlases`) are versioned tarballs
  with reference, annotation and structure tree. `neurofaune.atlas.brainglobe.BrainGlobeAtlas`
  reads them as RAS NIfTI.

## In use (built into the hub)

| atlas | what | where on disk | used for |
|---|---|---|---|
| **Antwerp C57BL/6 MRI** (UAntwerpen) | in vivo T2w head template + 46 MRI-drawn labels | `/mnt/arborea/atlases/Antwerp/` (from source-localization `data/atlas/`) | hub space shared with EEG |
| **Allen CCFv3** (Wang et al. 2020, *Cell*) | the reference annotation (structure IDs) | BrainGlobe `allen_mouse_25um` v1.2 in `/mnt/arborea/atlases/brainglobe/` | label definitions (allen32/26/64) |
| **Perens et al. 2023** (*Neuroinformatics*, doi 10.1007/s12021-023-09623-9) | in situ T2w MRI template with CCFv3 labels, LSFM template, skull-derived stereotaxic coordinates (micro-CT bregma), deformation fields MRI↔CCFv3↔LSFM | BrainGlobe `perens_stereotaxic_mri_mouse_25um` v1.0, plus the original release in `/mnt/arborea/atlases/perens_2023/` (neuropedia.dk, browser download only) | CCFv3 → Antwerp bridge; bregma |
| **RABIES priors** (Desrosiers-Grégoire et al. 2024, *Nat Commun* 15:6708) | 30 group-ICA rs-fMRI components on the RABIES EPI template; BOLD = IC 5, 12, 19 | `/mnt/arborea/atlases/rabies_extras_5118030/` (Zenodo 10.5281/zenodo.5118030, CC-BY-4.0) | functional layer |

## Functional candidates

| resource | what | space / format | status | notes for implementation |
|---|---|---|---|---|
| **Grandjean et al. 2020** "Common functional networks in the mouse brain revealed by multi-centre resting-state fMRI analysis", *NeuroImage* 205:116278 | 17 datasets, one pipeline; reproducible networks incl. a mouse DMN | data: OpenNeuro **ds001720** (BIDS, "Mouse_rest_multicentre"); preprocessed resource on Radboud/Donders (`data.ru.nl` collection DSC_4180000.18_502): AIBS template at 200 µm, AIBS-derived atlas time series | no network maps released that we found | Could build our own group ICA in Antwerp space from it. Network specificity depended on field strength, cryoprobe, ventilation and medetomidine-isoflurane, so expect weaker networks on isoflurane + volume coil. |
| **Gozzi lab, triple-network** (Mandino et al. 2022, *Mol Psychiatry*, doi 10.1038/s41380-021-01298-5) | mouse DMN, salience and lateral cortical networks | preprocessed data public (DOI repository); code as a GitLab R notebook | network maps not confirmed as a ready download | A natural follow-up if we name networks beyond the RABIES three. |
| **Allen connectivity modules** (Harris et al. 2019, *Nature* 575:195) | cortical areas grouped into modules from tracer connectivity: prefrontal, anterolateral, somatomotor, visual, medial, auditory, from memory | lists of CCFv3 areas (supplementary tables) | **membership not yet verified**; needs the paper's supplementary table | Cheapest option: a label grouping on the CCFv3 annotation we already have, so a coarse "network" grouping of allen26/32 with no registration. Zingg et al. 2014 (*Cell*) cortical subnetworks are an alternative grouping. |
| **Widefield calcium parcellations**, e.g. LocaNMF (Saxena et al. 2020, *PLoS Comput Biol*) | data-driven functional regions of dorsal cortex, per mouse, aligned to CCF | dorsal cortex only | methods, not a downloadable atlas | Relevant to EEG, since dorsal cortex is what the skull array sees, but would need widefield data. |

## Anatomical / multimodal candidates

| atlas | what | where / status | notes |
|---|---|---|---|
| **DSURQE** (Dorr–Steadman–Ullmann–Richards–Qiu–Egan; MICe Toronto) | 40 µm ex vivo MRI average, 336 labels; the standard RABIES/MICe template | **`repo.mouseimaging.ca` no longer resolves.** Dorr 2008 is on BrainGlobe as `dorr_mouse_mri_32um` v1.0, **downloaded** to `/mnt/arborea/atlases/brainglobe/`. The RABIES EPI template's 337 labels are probably DSURQE-derived. | Candidate second bridge to cross-validate the Perens route. GIN mirrors of the DSURQE notes exist (Aswendt lab). |
| **BlueBrain CCFv3 augmented** (Piluso et al. 2025) | extended, improved CCFv3 annotation and Nissl volume covering the entire brain | BrainGlobe `ccfv3augmented_mouse_{10,25}um`; Zenodo 15176439; GitHub `BlueBrain/ccfv3a-extended-atlas` | Fixes CCFv3 gaps (whole brain, cerebellar layers). Would replace CCFv3 IDs in the parcellations. |
| **Duke Mouse Brain Atlas** (Science Advances 2024, doi 10.1126/sciadv.adq8089) | MRI + LSFM **stereotaxic** atlas; bregma from projected sagittal/coronal suture fits | publication resources | An independent skull-derived bregma to cross-check Perens. |
| **Australian Mouse Brain Mapping Consortium (AMBMC)** | 15 µm ex vivo MRI model brain | BrainGlobe `australian_mouse_15um` | Higher-resolution MRI template; another bridge candidate. |
| **Kim / Osten / Princeton mouse atlases** | LSFM or STPT templates in CCF-derived labels (Kim isotropic, Osten STPT, Princeton LSFM) | BrainGlobe `kim_mouse_*`, `osten_mouse_*`, `princeton_mouse_20um` | Mainly for light-sheet/histology alignment, less for MRI. |
| **DevCCF** (Kronman et al. 2024, *Nat Commun*) and Kim developmental MRI/LSFM | developmental mouse templates (E11.5–P56), MRI contrasts incl. T2, ADC, FA | BrainGlobe `kim_dev_mouse_*` | If future cohorts are young animals. |
| **ANTsX mouse mapping** (Tustison et al. 2024, ANTsX ecosystem for mapping the mouse brain) | ANTs pipelines and transforms linking DevCCF/CCFv3 | paper + code | Method reference for MRI↔CCF registration. |
| **brainglobe-ccf-translator** | coordinate and volume translation between atlas spaces (CCFv3 among them; not DSURQE) | PyPI | Could replace custom point transforms between registered spaces. |

## Suggested order, if picked up

1. **Allen connectivity modules** as a grouping of allen26/32. No registration needed; first
   verify the membership table.
2. **Dorr / DSURQE as a second bridge.** Already downloaded. It cross-validates the Perens
   labels (inter-route Dice per parcel) in both projects.
3. **BlueBrain CCFv3 augmented labels**, to fill CCFv3's coverage gaps before any new
   source-localization head model.
4. **A group ICA from Grandjean ds001720** in Antwerp space. It gives a functional layer
   computed on our own template instead of borrowing RABIES priors. Large download.
5. **Duke stereotaxic bregma**, as an independent check on the Perens-derived bregma.
