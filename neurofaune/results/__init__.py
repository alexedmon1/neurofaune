"""The results specification: what an analysis writes so anyone can read it.

See docs/RESULTS_SPEC.md (the specification) and docs/RESULTS_PRODUCERS.md (how
neurofaune, neurovrai and study scripts meet it). Standard library only, so the
package can run -- or be copied -- where neurofaune's imaging stack is absent:
``python -m neurofaune.results check <folder>``.
"""
from .check import Report, check, check_analysis, find_analyses, read_table
from .spec import SPEC, SPEC_VERSION, STANDARD_TERMS
from .write import (NonConformingResults, columns_for, provenance,
                    write_analysis, write_columns)

__all__ = ["SPEC", "SPEC_VERSION", "STANDARD_TERMS", "Report", "check", "check_analysis",
           "find_analyses", "read_table", "NonConformingResults", "columns_for", "provenance",
           "write_analysis", "write_columns"]
