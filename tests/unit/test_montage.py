"""Montages of an analysis folder's maps (neurofaune.analysis.stats.montage): display only."""
import json

import nibabel as nib
import numpy as np
from click.testing import CliRunner


def _readout(tmp_path, with_mask=True):
    from neurofaune.analysis.stats import readout as ro
    from neurofaune.analysis.stats.readout_results import write_readout_results

    rng = np.random.default_rng(1)
    shape, aff = (8, 6, 7), np.eye(4)
    mask = np.zeros(shape, bool)
    mask[1:7, 1:5, 1:6] = True
    data = rng.normal(0.5, 1, shape + (10,)) * mask[..., None]
    rd = tmp_path / "an" / "randomise_FA"
    rd.mkdir(parents=True)
    tstat = np.where(mask, rng.normal(0, 2, shape), 0)
    corrp = np.where(mask & (np.abs(tstat) > 2), 0.99, 0.2)
    for name, img in (("data", data), ("mask", mask.astype(float)), ("randomise_tstat1", tstat),
                      ("randomise_tfce_corrp_tstat1", corrp), ("randomise_tfce_p_tstat1", corrp)):
        nib.save(nib.Nifti1Image(np.asarray(img, np.float32), aff), str(rd / f"{name}.nii.gz"))
    (rd / "design.mat").write_text("/NumWaves 1\n/NumPoints 10\n/Matrix\n" + "1\n" * 10)
    (rd / "design.con").write_text("/ContrastName1 mean>0\n/NumWaves 1\n/NumContrasts 1\n/Matrix\n1\n")
    atlas = ro.Atlas(mask.astype(np.int32), {1: "Region"}, {1: "L"})
    tests, clusters = ro.read_randomise(rd, rd / "data.nii.gz", rd / "design.mat", rd / "mask.nii.gz",
                                        atlas=atlas, labels={"metric": "FA"})
    rep = write_readout_results(
        tmp_path / "an", tests, clusters, analysis_id="dwi/voxelwise/one", title="one", description="one sample",
        analysis_type="voxelwise", modality="dwi", measure_column="metric", measures=["FA"], run_dirs={"FA": rd},
        n_permutations=10, alpha=0.05, mask_name="brain mask", space="test", inference="3-D TFCE",
        mask=rd / "mask.nii.gz" if with_mask else None, axes="RSA", strict=True)
    return tmp_path / "an", rep


def test_a_readout_draws_its_montage_and_still_conforms(tmp_path):
    folder, rep = _readout(tmp_path)
    assert rep.ok, rep.errors
    figures = json.loads((folder / "analysis.json").read_text())["figures"]
    (fig,) = [f for f in figures if f["path"].startswith("figures/montage__")]
    assert fig["path"] == "figures/montage__FA.png" and fig["measure"] == "FA"
    assert "corrected p < 0.05" in fig["caption"] and "outline: uncorrected" in fig["caption"]
    assert (folder / fig["path"]).stat().st_size > 1000


def test_no_mask_no_montage_and_rerunning_replaces(tmp_path):
    from neurofaune.analysis.stats.montage import add_montages

    folder, rep = _readout(tmp_path, with_mask=False)
    assert rep.ok and not json.loads((folder / "analysis.json").read_text()).get("figures")
    folder2, _ = _readout(tmp_path / "b")
    add_montages(folder2)
    figures = json.loads((folder2 / "analysis.json").read_text())["figures"]
    assert sum(f["path"] == "figures/montage__FA.png" for f in figures) == 1


def test_an_older_folder_gets_its_montage_without_newer_terms(tmp_path):
    """A 0.3 folder (cuprizone's TBSS trial) may not use figures[].kind: it is left out."""
    from neurofaune.analysis.stats.montage import add_montages
    from neurofaune.results import check_analysis

    folder, _ = _readout(tmp_path)
    meta = json.loads((folder / "analysis.json").read_text())
    meta["spec_version"], meta["figures"] = "0.3.0", []
    (folder / "analysis.json").write_text(json.dumps(meta))
    before = check_analysis(folder).errors
    (fig,) = add_montages(folder)
    assert "kind" not in fig
    assert check_analysis(folder).errors == before  # the montage adds nothing the version forbids


def test_the_montage_command(tmp_path):
    from neurofaune.preprocess.cli import main

    folder, _ = _readout(tmp_path)
    out = CliRunner().invoke(main, ["results", "montage", str(tmp_path)])
    assert out.exit_code == 0, out.output
    assert "OK   1 montage(s)" in out.output
