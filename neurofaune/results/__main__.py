"""``python -m neurofaune.results check <path> [--json]`` -- neurobox's conformance checker."""
import sys

from neurofaune.results import AVAILABLE, INSTALL_HINT

if AVAILABLE:
    from neurobox.__main__ import main, run_check  # noqa: F401
else:
    def run_check(path: str, as_json: bool = False) -> int:
        print(f"cannot check {path}: {INSTALL_HINT}", file=sys.stderr)
        return 2

    def main(argv: list[str] | None = None) -> int:
        print(INSTALL_HINT, file=sys.stderr)
        return 2

if __name__ == "__main__":
    sys.exit(main())
