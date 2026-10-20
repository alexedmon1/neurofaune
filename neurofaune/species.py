"""Species profiles: the layer between package defaults and a study config.

``configs/default.yaml`` is tuned for rat -- SIGMA, a 10x voxel scale chosen
for rat brain size, brain-volume QC ranges in rat mm3, skull-strip radii for a
rat brain. A mouse brain is about a quarter of the volume, so those values are
wrong for it, not merely suboptimal.

Rather than fork the defaults, each species has a profile
(``configs/species/<name>.yaml``) that overrides only what differs. The merge
order is::

    configs/default.yaml  <  configs/species/<species>.yaml  <  study config.yaml

so a study can still override anything its profile sets. The species is read
from the study config's ``study.species`` (default ``rat``). The rat profile is
empty by design: the defaults *are* the rat values, so selecting rat changes
nothing, and existing studies keep producing identical outputs.
"""

from pathlib import Path
from typing import Any, Dict, List

import yaml

#: Species used when a study config does not name one. Every study that
#: predates species profiles is a rat study.
DEFAULT_SPECIES = "rat"

SPECIES_DIR = Path(__file__).parent.parent / "configs" / "species"


class UnknownSpeciesError(ValueError):
    """Raised when ``study.species`` names a species with no profile."""


def available_species() -> List[str]:
    """Names of the species that have a profile."""
    return sorted(p.stem for p in SPECIES_DIR.glob("*.yaml"))


def species_of(study_config: Dict[str, Any]) -> str:
    """The species a study config selects (lower-cased), or the default."""
    species = (study_config.get("study") or {}).get("species") or DEFAULT_SPECIES
    return str(species).strip().lower()


def load_species_profile(species: str) -> Dict[str, Any]:
    """Load the override profile for ``species``.

    Raises
    ------
    UnknownSpeciesError
        If there is no profile. A typo here would otherwise silently fall back
        to rat parameters, which is exactly the failure this layer exists to
        prevent.
    """
    path = SPECIES_DIR / f"{species}.yaml"
    if not path.exists():
        raise UnknownSpeciesError(
            f"No species profile for {species!r} (looked for {path}). "
            f"Available: {', '.join(available_species())}"
        )
    with open(path) as f:
        profile = yaml.safe_load(f) or {}
    return profile


def apply_species_profile(base: Dict[str, Any], study_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge ``base`` < species profile < ``study_config``.

    ``base`` is normally the package defaults; it may be ``{}`` for callers
    that read a study YAML directly (e.g. ``neurofaune bids``) and only need
    the profile's values under the study's.
    """
    from neurofaune.config import merge_configs

    profile = load_species_profile(species_of(study_config))
    return merge_configs(merge_configs(base, profile), study_config)
