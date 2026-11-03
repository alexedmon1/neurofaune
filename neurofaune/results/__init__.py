"""The results specification, re-exported from the optional neurobox (neurobox-dev).

It moved out of neurofaune on 2026-11-03, unchanged, so tools other than neurofaune can
use it; the specification is neurobox-dev's docs/RESULTS_SPEC.md. It is **optional**:
neurofaune installs and runs without it, and with it (``pip install 'neurofaune[results]'``)
analyses also write analysis.json, provenance.json and column dictionaries, and
``neurofaune results check`` works. ``AVAILABLE`` says which.

When installed, every ``neurofaune.results`` import keeps working: each
``neurofaune.results.<module>`` *is* the neurobox module (one object, so patches and
identity checks agree), and ``neurofaune.results.check`` is still the function. Like
neurobox, this loads nothing but the standard library.
"""
import importlib as _importlib
import sys as _sys

INSTALL_HINT = ("the optional results package neurobox-dev is not installed; "
                "install it with: pip install 'neurofaune[results]' (or: uv sync --extra results)")

try:
    import neurobox as _neurobox  # noqa: F401
except ImportError:
    AVAILABLE = False

    def __getattr__(name):
        raise ImportError(f"neurofaune.results.{name}: {INSTALL_HINT}")
else:
    AVAILABLE = True
    from neurobox import *  # noqa: F401,F403
    from neurobox import __all__  # noqa: F401

    for _name in ("_schema", "check", "spec", "write"):
        _sys.modules[f"{__name__}.{_name}"] = _importlib.import_module(f"neurobox.{_name}")
    _schema = _sys.modules[f"{__name__}._schema"]
    spec = _sys.modules[f"{__name__}.spec"]
    write = _sys.modules[f"{__name__}.write"]
    # `check` stays the function `from neurobox import *` bound, as it was before the move;
    # the module is `neurofaune.results.check` in sys.modules, for `from ... import`.
