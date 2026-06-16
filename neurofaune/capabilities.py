"""neurofaune capability introspection.

Single source of truth for *what neurofaune can do*, generated from the code so
it cannot drift. Walks the public subpackages, collects entry-point functions
(``run_*``, ``build_*``, ``register_*``, ``propagate_*``, ``fit_*``,
``compute_*``, ``extract_*``, ``segment_*``, ``select_*``), and for each records
its module, one-line docstring summary, and the config keys it reads.

Used by:
- ``neurofaune capabilities`` (CLI) — print the catalog (text / markdown / json).
- ``make capabilities`` — regenerate ``CAPABILITIES.md`` at the repo root.
- ``tests/unit/test_capabilities.py`` — assert the committed catalog is current,
  so any new entry point must be cataloged (the make-check gate).

The point: discovery (what exists) and upgrade-drift (what changed between pins)
both reduce to reading / diffing the generated catalog.
"""
from __future__ import annotations

import importlib
import inspect
import pkgutil
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Entry-point name prefixes that denote a public capability (vs. a helper).
ENTRY_PREFIXES = (
    "run_", "build_", "register_", "propagate_", "fit_", "compute_",
    "extract_", "segment_", "select_", "preprocess_", "convert_",
    "denoise_", "degibbs_", "warp_", "normalize_",
)

# Modules excluded from the scan (not capability surface).
EXCLUDE_SUBSTRINGS = ("capabilities", ".cli", ".tests", "._")

# More specific module-prefix → stage label, checked in order (first match wins).
# A fallback derives the stage from the package path, so a NEW subpackage is
# cataloged automatically — the scan is complete by construction, not by a
# hand-maintained list (which would itself be a "things get missed" risk).
STAGE_PREFIXES = (
    ("neurofaune.preprocess.workflows", "preprocess (workflows)"),
    ("neurofaune.preprocess.qc", "preprocess (qc)"),
    ("neurofaune.preprocess.utils", "preprocess (utils)"),
    ("neurofaune.preprocess", "preprocess"),
)

_CONFIG_KEY_RE = re.compile(r"""get_config_value\(\s*[A-Za-z_][\w]*\s*,\s*['"]([\w.]+)['"]""")


@dataclass
class Capability:
    name: str
    module: str
    stage: str
    summary: str
    config_keys: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "module": self.module,
            "stage": self.stage,
            "summary": self.summary,
            "config_keys": self.config_keys,
        }


def _first_doc_line(obj) -> str:
    doc = inspect.getdoc(obj) or ""
    for line in doc.splitlines():
        line = line.strip()
        if line:
            return line
    return "(no docstring)"


def _config_keys(func) -> List[str]:
    try:
        src = inspect.getsource(func)
    except (OSError, TypeError):
        return []
    return sorted(set(_CONFIG_KEY_RE.findall(src)))


# Modules that failed to import during the last collect_capabilities() walk.
# Surfaced in the catalog so a capability hidden behind an import error is
# visible, not silently dropped (the exact failure mode this tool prevents).
IMPORT_ERRORS: Dict[str, str] = {}


def _iter_modules(pkg_name: str):
    """Yield the package and all of its submodules, recursively. Import failures
    are recorded in IMPORT_ERRORS (and the offending module skipped) rather than
    silently swallowed."""
    try:
        pkg = importlib.import_module(pkg_name)
    except Exception as e:  # noqa: BLE001
        IMPORT_ERRORS[pkg_name] = f"{type(e).__name__}: {e}"
        return
    yield pkg
    if not hasattr(pkg, "__path__"):
        return

    def _onerror(name: str) -> None:
        # walk_packages calls this when a submodule import raises
        import sys
        exc = sys.exc_info()[1]
        IMPORT_ERRORS[name] = f"{type(exc).__name__}: {exc}" if exc else "import error"

    for info in pkgutil.walk_packages(pkg.__path__, prefix=pkg_name + ".", onerror=_onerror):
        if any(part.startswith("_") for part in info.name.split(".")):
            continue
        try:
            yield importlib.import_module(info.name)
        except Exception as e:  # noqa: BLE001
            IMPORT_ERRORS[info.name] = f"{type(e).__name__}: {e}"
            continue


def _stage_for(module_name: str) -> str:
    """Derive a stage label from a module path. Specific prefixes win; otherwise
    fall back to the package immediately under `neurofaune` so a brand-new
    subpackage is labeled (and cataloged) without touching this file."""
    for prefix, label in STAGE_PREFIXES:
        if module_name == prefix or module_name.startswith(prefix + "."):
            return label
    parts = module_name.split(".")
    return parts[1] if len(parts) > 1 else module_name


def collect_capabilities() -> List[Capability]:
    """Walk the entire neurofaune package and return all entry-point capabilities,
    sorted deterministically by (stage, module, name) so the generated catalog is
    stable. Completeness is by construction: every importable module is scanned."""
    IMPORT_ERRORS.clear()
    caps: Dict[str, Capability] = {}
    for mod in _iter_modules("neurofaune"):
        if any(sub in mod.__name__ for sub in EXCLUDE_SUBSTRINGS):
            continue
        for name, func in inspect.getmembers(mod, inspect.isfunction):
            if not name.startswith(ENTRY_PREFIXES):
                continue
            # only functions DEFINED in this module (not imported into it)
            if getattr(func, "__module__", None) != mod.__name__:
                continue
            key = f"{mod.__name__}.{name}"
            if key in caps:
                continue
            caps[key] = Capability(
                name=name,
                module=mod.__name__,
                stage=_stage_for(mod.__name__),
                summary=_first_doc_line(func),
                config_keys=_config_keys(func),
            )
    return sorted(caps.values(), key=lambda c: (c.stage, c.module, c.name))


def cli_commands() -> List[str]:
    """Names of registered `neurofaune` CLI subcommands."""
    try:
        from neurofaune.preprocess.cli import main
        return sorted(main.commands.keys())
    except Exception:  # noqa: BLE001
        return []


def _version() -> str:
    try:
        from importlib.metadata import version
        return version("neurofaune")
    except Exception:  # noqa: BLE001
        return "unknown"


def render_markdown() -> str:
    caps = collect_capabilities()
    cmds = cli_commands()
    lines = [
        "# neurofaune capabilities",
        "",
        f"_Generated from the code by `neurofaune capabilities` (v{_version()})._ "
        "Do not edit by hand — run `make capabilities`.",
        "",
        f"**CLI subcommands:** {', '.join('`'+c+'`' for c in cmds) or '(none)'}",
        "",
        f"**Entry points:** {len(caps)} across {len({c.stage for c in caps})} stages.",
        "",
    ]
    stage = None
    for c in caps:
        if c.stage != stage:
            stage = c.stage
            lines += ["", f"## {stage}", ""]
            lines.append("| function | module | summary | config keys |")
            lines.append("|---|---|---|---|")
        ck = ", ".join(f"`{k}`" for k in c.config_keys) or "—"
        summary = c.summary.replace("|", "\\|")
        lines.append(f"| `{c.name}` | `{c.module}` | {summary} | {ck} |")
    if IMPORT_ERRORS:
        lines += ["", "## ⚠ import errors (capabilities may be hidden)", ""]
        for mod in sorted(IMPORT_ERRORS):
            lines.append(f"- `{mod}` — {IMPORT_ERRORS[mod]}")
    return "\n".join(lines) + "\n"


def render_text() -> str:
    caps = collect_capabilities()
    out = [f"neurofaune v{_version()} — {len(caps)} capabilities",
           f"CLI: {', '.join(cli_commands()) or '(none)'}", ""]
    stage = None
    for c in caps:
        if c.stage != stage:
            stage = c.stage
            out.append(f"\n[{stage}]")
        out.append(f"  {c.name:34s} {c.summary}")
    return "\n".join(out) + "\n"


def render_json() -> str:
    import json
    return json.dumps(
        {"version": _version(), "cli": cli_commands(),
         "capabilities": [c.to_dict() for c in collect_capabilities()]},
        indent=2,
    ) + "\n"
