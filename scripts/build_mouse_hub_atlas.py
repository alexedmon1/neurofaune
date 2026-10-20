#!/usr/bin/env python3
"""Build the mouse hub atlas (Antwerp space, CCFv3 labels via the Perens T2w bridge).

Example wrapper around neurofaune.atlas.antwerp.build_antwerp_hub; the logic
lives in the module. Also writes the old-vs-new per-parcel comparison and the
rostro-caudal landmark scale check into <out>/comparison/.

    uv run python scripts/build_mouse_hub_atlas.py \\
        --source-localization-atlas ~/sandbox/source-localization/src/source_localization/data/atlas \\
        --perens /mnt/arborea/atlases/brainglobe/perens_stereotaxic_mri_mouse_25um_v1.0.extract/perens_stereotaxic_mri_mouse_25um_v1.0 \\
        --out /mnt/arborea/atlases/Antwerp --work ~/work/antwerp_build
"""
import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from neurofaune.atlas.antwerp import build_antwerp_hub
from neurofaune.atlas.brainglobe import BrainGlobeAtlas
from neurofaune.atlas.compare import ap_landmark_scale, compare_parcellations

#: Old source-localization volumes name the hippocampal halves the wrong way
#: round; re-map them by position before comparing.
OLD_HIPPOCAMPUS_SWAP = {"allen32": {7: 8, 8: 7, 23: 24, 24: 23}, "allen26": {4: 5, 5: 4, 14: 15, 15: 14}}
LANDMARKS = {"MOB": "MOB", "ac": "act", "pc": "pc", "IC": "IC", "CB": "CB"}
PAIRS = [("MOB", "CB"), ("ac", "pc"), ("ac", "IC"), ("MOB", "IC")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-localization-atlas", type=Path, required=True)
    ap.add_argument("--perens", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    sl = a.source_localization_atlas
    maps = {"allen32": sl / "allen/roi_mapping.json", "allen26": sl / "allen/roi_mapping_allen26.json",
            "allen64": sl / "allen/roi_mapping_allen64.json"}
    meta = build_antwerp_hub(sl, a.perens, a.out, a.work, maps, sl / "roi_mapping.json", n_threads=a.threads)

    comp = a.out / "comparison"
    comp.mkdir(exist_ok=True)
    perens = BrainGlobeAtlas(a.perens)
    ids = {k: perens.descendants(perens.by_acronym(v)) for k, v in LANDMARKS.items()}
    native = perens.annotation()
    scale = {"new": ap_landmark_scale(native, nib.load(str(a.out / "ccfv3_annotation.nii.gz")), ids, PAIRS),
             "old": ap_landmark_scale(native, nib.load(str(sl / "allen/allen_annotation_in_antwerp.nii.gz")), ids, PAIRS)}
    out = {"ap_landmark_scale": scale}
    for name, old_file in (("allen32", "allen/allen_labels.nii.gz"), ("allen26", "allen/allen26_labels.nii.gz")):
        new = nib.load(str(a.out / f"{name}_labels.nii.gz"))
        od = np.asanyarray(nib.load(str(sl / old_file)).dataobj).astype(np.int64)
        fixed = od.copy()
        for x, y in OLD_HIPPOCAMPUS_SWAP[name].items():
            fixed[od == x] = y
        rois = json.loads((a.out / f"roi_mapping_{name}.json").read_text())["rois"]
        rows = compare_parcellations(nib.Nifti1Image(fixed.astype(np.uint16), new.affine), new,
                                     {int(k): v["name"] for k, v in rois.items()})
        with open(comp / f"{name}_old_vs_new.tsv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
            w.writeheader()
            w.writerows(rows)
        out[name] = rows
    (comp / "summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(meta["validation"]["median_composite_dice"]))


if __name__ == "__main__":
    main()
