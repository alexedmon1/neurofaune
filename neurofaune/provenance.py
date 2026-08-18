"""Record which neurofaune, and which settings, produced a derivative.

A derivative that cannot name the code that made it cannot be reproduced, and
the question comes up precisely when it is hardest to answer -- at a freeze, or
when two cohorts disagree. On the cuprizone study the four arms were each built
under a different pin and patched forward with regens, and recovering which was
which meant reading file timestamps against a hand-written activity log, because
nothing in ``derivatives/`` recorded a version.

The exact commit IS recoverable from an installed package, without a git
checkout: pip and uv write PEP 610 ``direct_url.json`` into the ``.dist-info``
of anything installed from a VCS, and it carries both the resolved commit and
the ref that was asked for. That is what :func:`package_provenance` reads.

Output follows the BIDS ``GeneratedBy`` convention so the records are meaningful
to tools other than this one.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PACKAGE = "neurofaune"


def package_provenance() -> Dict[str, Any]:
    """Identify the running neurofaune as precisely as the install allows.

    ``Version`` is always present. ``CommitID``/``RequestedRevision`` appear when
    the package was installed from a VCS -- which is how studies are expected to
    pin it -- and are absent for an editable or PyPI install, rather than being
    guessed.
    """
    import importlib.metadata as md

    entry: Dict[str, Any] = {"Name": PACKAGE}
    try:
        dist = md.distribution(PACKAGE)
        entry["Version"] = dist.version
    except md.PackageNotFoundError:          # pragma: no cover - not installed
        entry["Version"] = "unknown"
        return entry

    try:
        raw = dist.read_text("direct_url.json")
        if raw:
            info = json.loads(raw)
            entry["CodeURL"] = info.get("url")
            vcs = info.get("vcs_info") or {}
            if vcs.get("commit_id"):
                entry["CommitID"] = vcs["commit_id"]
            if vcs.get("requested_revision"):
                entry["RequestedRevision"] = vcs["requested_revision"]
    except Exception:                        # pragma: no cover - metadata varies
        pass
    return entry


def config_digest(config: Optional[Any]) -> Optional[str]:
    """Stable sha256 over a config.

    Catches the case the version alone misses: same code, different settings.
    Keys are sorted and values coerced to str so the digest does not move for
    reasons that are not real changes (dict ordering, Path vs str).
    """
    if config is None:
        return None
    try:
        canon = json.dumps(config, sort_keys=True, default=str)
    except TypeError:                        # pragma: no cover - exotic config
        return None
    return "sha256:" + hashlib.sha256(canon.encode()).hexdigest()


def generated_by(config: Optional[Any] = None) -> List[Dict[str, Any]]:
    """A BIDS ``GeneratedBy`` list for this run."""
    entry = package_provenance()
    entry["DateTime"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    digest = config_digest(config)
    if digest:
        entry["ConfigDigest"] = digest
    return [entry]


def stamp(metadata: Dict[str, Any], config: Optional[Any] = None) -> Dict[str, Any]:
    """Add ``GeneratedBy`` to a metadata dict, in place, and return it."""
    metadata["GeneratedBy"] = generated_by(config)
    return metadata


def write_provenance(
    derivatives_dir: Path,
    subject: str,
    session: str,
    modality: str,
    config: Optional[Any] = None,
    sources: Optional[List[Path]] = None,
) -> Path:
    """Write ``<sub>_<ses>_<mod>-provenance.json`` beside the derivatives.

    A separate file rather than a field on an existing sidecar, because the arms
    do not agree on having one: the anat workflow writes no JSON at all, so
    there is nothing to stamp. One file per session per modality is uniform
    across arms and survives an arm adding or removing sidecars later.
    """
    derivatives_dir = Path(derivatives_dir)
    derivatives_dir.mkdir(parents=True, exist_ok=True)
    record: Dict[str, Any] = {
        "Subject": subject,
        "Session": session,
        "Modality": modality,
        "GeneratedBy": generated_by(config),
    }
    if sources:
        record["Sources"] = [str(s) for s in sources]
    out = derivatives_dir / f"{subject}_{session}_{modality}-provenance.json"
    out.write_text(json.dumps(record, indent=2))
    return out


def write_dataset_description(
    derivatives_root: Path,
    name: str = "neurofaune preprocessing",
    config: Optional[Any] = None,
) -> Path:
    """Write the BIDS-derivatives ``dataset_description.json`` at the root.

    Rewritten on every run, so it names the version that produced the most
    recent output. Per-session provenance is the authority for any individual
    derivative -- a cohort processed across several versions will have a root
    description matching only the last one.
    """
    derivatives_root = Path(derivatives_root)
    derivatives_root.mkdir(parents=True, exist_ok=True)
    out = derivatives_root / "dataset_description.json"
    out.write_text(json.dumps({
        "Name": name,
        "BIDSVersion": "1.9.0",
        "DatasetType": "derivative",
        "GeneratedBy": generated_by(config),
    }, indent=2))
    return out
