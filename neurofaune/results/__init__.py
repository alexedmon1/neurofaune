"""The results specification, re-exported from the optional Lumière (``lumiere``).

It moved out of neurofaune on 2026-11-03, unchanged, so tools other than neurofaune can
use it; the specification is Lumière's docs/RESULTS_SPEC.md (github.com/alexedmon1/lumiere,
called neurobox-dev until 2026-11-10). It is **optional**: neurofaune installs and runs
without it, and with it (``pip install 'neurofaune[results]'``) analyses also write
analysis.json, provenance.json and column dictionaries, ``neurofaune results check`` works and
``neurofaune gallery build`` builds a study's gallery. ``AVAILABLE`` says which.

When installed, every ``neurofaune.results`` import keeps working: each
``neurofaune.results.<module>`` *is* the Lumière module (one object, so patches and
identity checks agree), and ``neurofaune.results.check`` is still the function. Like
Lumière, this loads nothing but the standard library.
"""
import importlib as _importlib
import sys as _sys

INSTALL_HINT = ("the optional results package Lumière (lumiere) is not installed; "
                "install it with: pip install 'neurofaune[results]' (or: uv sync --extra results)")

#: Lumière's modules, each also importable as ``neurofaune.results.<name>``.
SUBMODULES = ("_schema", "check", "spec", "write", "gallery", "gallery.digest", "gallery.links")

try:
    import lumiere as _lumiere  # noqa: F401
except ImportError:
    AVAILABLE = False

    def __getattr__(name):
        raise ImportError(f"neurofaune.results.{name}: {INSTALL_HINT}")
else:
    AVAILABLE = True
    from lumiere import *  # noqa: F401,F403
    from lumiere import __all__  # noqa: F401

    for _name in SUBMODULES:
        _sys.modules[f"{__name__}.{_name}"] = _importlib.import_module(f"lumiere.{_name}")
    _schema = _sys.modules[f"{__name__}._schema"]
    spec = _sys.modules[f"{__name__}.spec"]
    write = _sys.modules[f"{__name__}.write"]
    gallery = _sys.modules[f"{__name__}.gallery"]
    # `check` stays the function `from lumiere import *` bound, as it was before the move;
    # the module is `neurofaune.results.check` in sys.modules, for `from ... import`.
