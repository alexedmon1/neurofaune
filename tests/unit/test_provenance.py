"""A derivative must be able to name the code and settings that produced it.

On the cuprizone cohort the four arms were each built under a different pin and
patched forward with regens; recovering which was which meant reading file
mtimes against a hand-written log, because nothing in derivatives/ recorded a
version. These tests pin the parts that make that recoverable from the data.
"""
import json

from neurofaune import provenance


def test_version_is_always_recorded():
    e = provenance.package_provenance()
    assert e["Name"] == "neurofaune"
    assert e["Version"] and e["Version"] != "unknown"


def test_vcs_install_records_the_exact_commit(monkeypatch):
    """A pinned study install must be traceable to one commit, not just a tag.

    Tags can move; the resolved commit cannot. PEP 610 direct_url.json carries
    both, so record both.
    """
    class FakeDist:
        version = "0.6.3a0"

        def read_text(self, name):
            assert name == "direct_url.json"
            return json.dumps({
                "url": "https://github.com/alexedmon1/neurofaune.git",
                "vcs_info": {"vcs": "git", "commit_id": "abc123",
                             "requested_revision": "v0.6.3-alpha"},
            })

    monkeypatch.setattr("importlib.metadata.distribution", lambda n: FakeDist())
    e = provenance.package_provenance()
    assert e["CommitID"] == "abc123"
    assert e["RequestedRevision"] == "v0.6.3-alpha"
    assert e["CodeURL"].endswith("neurofaune.git")


def test_non_vcs_install_omits_commit_rather_than_guessing(monkeypatch):
    class FakeDist:
        version = "0.6.3a0"

        def read_text(self, name):
            return None

    monkeypatch.setattr("importlib.metadata.distribution", lambda n: FakeDist())
    e = provenance.package_provenance()
    assert "CommitID" not in e and "RequestedRevision" not in e
    assert e["Version"] == "0.6.3a0"


def test_config_digest_catches_settings_drift():
    """Same code, different settings is the case a version alone cannot catch."""
    a = {"functional": {"acompcor": {"variance_threshold": 0.9}}}
    b = {"functional": {"acompcor": {"variance_threshold": 0.5}}}
    assert provenance.config_digest(a) != provenance.config_digest(b)


def test_config_digest_is_stable_across_key_order_and_paths():
    """The digest must not move for reasons that are not real changes."""
    from pathlib import Path
    a = {"b": 2, "a": Path("/x")}
    b = {"a": Path("/x"), "b": 2}
    assert provenance.config_digest(a) == provenance.config_digest(b)
    assert provenance.config_digest(None) is None


def test_write_provenance_is_uniform_across_arms(tmp_path):
    """The anat arm writes no sidecar to stamp, so provenance is its own file."""
    out = provenance.write_provenance(
        tmp_path / "anat", "sub-1X", "ses-1", "anat",
        config={"a": 1}, sources=[tmp_path / "in.nii.gz"])
    d = json.loads(out.read_text())
    assert out.name == "sub-1X_ses-1_anat-provenance.json"
    assert d["Modality"] == "anat" and d["Subject"] == "sub-1X"
    assert d["GeneratedBy"][0]["Name"] == "neurofaune"
    assert d["GeneratedBy"][0]["ConfigDigest"].startswith("sha256:")
    assert d["Sources"][0].endswith("in.nii.gz")


def test_stamp_adds_generated_by_in_place():
    m = {"subject": "sub-1X"}
    assert provenance.stamp(m) is m
    assert m["GeneratedBy"][0]["Name"] == "neurofaune"
    assert m["subject"] == "sub-1X"


def test_dataset_description_is_bids_derivative(tmp_path):
    out = provenance.write_dataset_description(tmp_path, config={"a": 1})
    d = json.loads(out.read_text())
    assert d["DatasetType"] == "derivative"
    assert d["GeneratedBy"][0]["Version"]
