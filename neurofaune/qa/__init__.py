"""Static checks a study repo can run against its own code.

Not QC (that measures images). These check the *code* around the pipeline for
mistakes that produce plausible-looking output rather than an error.
"""
from neurofaune.qa.path_hygiene import (
    Finding,
    find_hardcoded_sigma_paths,
    scan_source,
)

__all__ = ["Finding", "find_hardcoded_sigma_paths", "scan_source"]
