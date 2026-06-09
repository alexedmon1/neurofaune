#!/usr/bin/env python3
"""neurofaune command-line interface.

Wired as the ``neurofaune`` console entrypoint (``[project.scripts]``). Today it
exposes the config-driven Bruker→BIDS converter; more subcommands (preprocess
phases) can be added to the same group.
"""
from __future__ import annotations

import logging
from pathlib import Path

import click


def _version() -> str:
    try:
        from importlib.metadata import version
        return version("neurofaune")
    except Exception:  # noqa: BLE001
        return "unknown"


@click.group()
@click.version_option(version=_version(), prog_name="neurofaune")
@click.option("-v", "--verbose", is_flag=True, help="DEBUG logging")
def main(verbose: bool) -> None:
    """neurofaune — rodent MRI preprocessing & analysis."""
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(levelname)s %(message)s")


@main.command()
@click.option("--config", type=click.Path(exists=True, path_type=Path),
              help="study YAML containing a 'bids:' block")
@click.option("--raw", type=click.Path(exists=True, path_type=Path), help="raw Bruker root (overrides config)")
@click.option("--bids", type=click.Path(path_type=Path), help="BIDS output root (overrides config)")
@click.option("--session-regex", help="regex with named 'subject'/'session' groups (overrides config)")
@click.option("--relabel", multiple=True, help="session relabel, e.g. 1a=1b (repeatable)")
@click.option("--map", "maps", multiple=True,
              help="sequence map override, e.g. 'Bruker:T2S_EPI=func/bold' (repeatable)")
@click.option("--layout", type=click.Choice(["flat", "nested"]), help="discovery layout")
@click.option("--subject", "subjects", multiple=True, help="restrict to subject id(s), e.g. 1Y (repeatable)")
@click.option("--scans-only", is_flag=True, help="(re)write per-session scans.tsv only; no conversion")
@click.option("--dry-run", is_flag=True, help="discover + parse only; write nothing")
def bids(config, raw, bids, session_regex, relabel, maps, layout, subjects, scans_only, dry_run):
    """Convert a raw Bruker study to BIDS/NIfTI (config-driven)."""
    import yaml
    from neurofaune.utils.bids import BidsConfig, convert_study, discover_sessions

    cfg_dict = {}
    if config:
        cfg_dict = yaml.safe_load(config.read_text()) or {}
    b = dict(cfg_dict.get("bids", {}))
    if raw:
        b["raw_root"] = str(raw)
    if bids:
        b["bids_root"] = str(bids)
    if session_regex:
        b["session_regex"] = session_regex
    if layout:
        b["layout"] = layout
    if relabel:
        b["session_relabel"] = {**b.get("session_relabel", {}),
                                **dict(r.split("=", 1) for r in relabel)}
    if maps:
        sm = dict(b.get("sequence_map", {}))
        for m in maps:
            method, ms = m.split("=", 1)
            mod, suf = ms.split("/", 1)
            sm[method] = {"modality": mod, "suffix": suf}
        b["sequence_map"] = sm
    cfg_dict["bids"] = b

    cfg = BidsConfig.from_config(cfg_dict)
    subjects = set(subjects) or None

    if dry_run:
        sessions = discover_sessions(cfg)
        if subjects:
            sessions = [(d, m) for d, m in sessions if m["subject"] in {s.upper() for s in subjects}]
        click.echo(f"DRY-RUN: {len(sessions)} session(s) match:")
        for d, m in sessions:
            click.echo(f"  sub-{m['subject']} ses-{m['session']}  <- {d.name}")
        return

    results = convert_study(cfg, subjects=subjects, convert=not scans_only)
    total = sum(r["n_written"] for r in results)
    click.echo(f"\nDone: {len(results)} session(s), {total} image(s) written to {cfg.bids_root}")


if __name__ == "__main__":
    main()
