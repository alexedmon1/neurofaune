"""``python -m neurofaune.results check <path> [--json]`` -- the conformance checker."""
from __future__ import annotations

import argparse
import json
import sys

from .check import check
from .spec import SPEC, SPEC_VERSION


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m neurofaune.results",
                                 description=f"{SPEC} {SPEC_VERSION} conformance checker")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="check every analysis folder at or under a path")
    c.add_argument("path")
    c.add_argument("--json", action="store_true", help="one JSON report per folder")
    a = ap.parse_args(argv)
    return run_check(a.path, as_json=a.json)


def run_check(path: str, as_json: bool = False) -> int:
    reports = check(path)
    if as_json:
        print(json.dumps([r.as_dict() for r in reports], indent=2))
    else:
        if not reports:
            print(f"no analysis folders ({SPEC}) at or under {path}", file=sys.stderr)
        for r in reports:
            print(f"{'OK  ' if r.ok else 'FAIL'} {r.id or '?'}  ({r.folder})")
            for e in r.errors:
                print(f"     error: {e}")
            for w in r.warnings:
                print(f"     warning: {w}")
    return 0 if reports and all(r.ok for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
