#!/usr/bin/env python3
"""neurofaune command-line interface.

Wired as the ``neurofaune`` console entrypoint (``[project.scripts]``). Today it
exposes the config-driven Bruker→BIDS converter and a capabilities report; more
subcommands (preprocess phases) can be added to the same group.
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


@main.command()
@click.option("--format", "fmt", type=click.Choice(["text", "md", "json"]), default="text",
              help="output format (default: text)")
@click.option("--output", type=click.Path(path_type=Path), help="write to file instead of stdout")
def capabilities(fmt, output):
    """List everything neurofaune can do (generated from the code)."""
    from neurofaune import capabilities as cap
    render = {"text": cap.render_text, "md": cap.render_markdown, "json": cap.render_json}[fmt]
    text = render()
    if output:
        output.write_text(text)
        click.echo(f"wrote {output}")
    else:
        click.echo(text, nl=False)


@main.command("check-paths")
@click.argument("paths", nargs=-1, required=True,
                type=click.Path(exists=True, path_type=Path))
def check_paths(paths):
    """Fail if code hardcodes a tpl->SIGMA transform filename.

    ANTs names those files after the --output prefix it was given, so the name is
    a property of how the study ran registration, not a constant. Code that spells
    one out keeps working until the layout changes, then silently resolves nothing
    -- the step skips, the run exits 0, and the gap shows up only in a log. Point
    this at a study's own code (e.g. `neurofaune check-paths preprocessing/code`)
    and gate the pipeline on it.
    """
    from neurofaune.qa import find_hardcoded_sigma_paths

    findings = find_hardcoded_sigma_paths(paths)
    for f in findings:
        click.echo(str(f), err=True)
    if findings:
        click.echo(f"\n{len(findings)} hardcoded tpl->SIGMA path(s)", err=True)
        raise SystemExit(1)
    click.echo("no hardcoded tpl->SIGMA paths")


if __name__ == "__main__":
    main()


@main.command("check-deps")
@click.option("--group", "groups", multiple=True,
              help="feature area(s) to check, e.g. tractography (repeatable; default: all)")
@click.option("--config", type=click.Path(exists=True, path_type=Path),
              help="study YAML; honours tractography.mrtrix_bin when resolving tools")
@click.option("--no-versions", is_flag=True, help="skip version probing (faster)")
@click.option("--strict", is_flag=True, help="exit non-zero if any required tool is missing")
def check_deps(groups, config, no_versions, strict):
    """Report which external tools (FSL, ANTs, MRtrix3) are installed.

    `uv sync` installs the Python side only; the neuroimaging suites neurofaune
    drives cannot come from PyPI. Without this, a missing binary surfaces as a
    FileNotFoundError from inside a workflow, naming one command and offering no
    route to getting it. Run this first on a new machine:

        neurofaune check-deps                        # everything
        neurofaune check-deps --group tractography   # just MRtrix3 + FSL tracking
        neurofaune check-deps --strict               # CI gate
    """
    from neurofaune.utils.dependencies import (
        check_dependencies, format_report, missing_packages,
    )

    extra_paths = []
    if config:
        import yaml
        from neurofaune.config import get_config_value
        cfg = yaml.safe_load(config.read_text()) or {}
        configured = get_config_value(cfg, "tractography.mrtrix_bin", default=None)
        if configured:
            extra_paths.append(str(configured))

    results = check_dependencies(
        groups=list(groups) or None,
        probe_versions=not no_versions,
        extra_paths=extra_paths or None,
    )
    click.echo(format_report(results, color=True))

    if strict and missing_packages(results):
        raise SystemExit(1)
