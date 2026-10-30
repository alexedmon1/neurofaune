#!/usr/bin/env python3
"""
Configuration loader for rodent MRI preprocessing pipeline.

Handles:
- Loading YAML configuration files
- Merging study configs with defaults
- Environment variable substitution
- Configuration validation
- Rodent-specific parameters (atlases, slice definitions, age cohorts)
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing required parameters."""
    pass


def load_yaml(file_path: Path) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Parameters
    ----------
    file_path : Path
        Path to YAML file

    Returns
    -------
    dict
        Loaded configuration

    Raises
    ------
    ConfigurationError
        If file doesn't exist or YAML is invalid
    """
    if not file_path.exists():
        raise ConfigurationError(f"Configuration file not found: {file_path}")

    try:
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
        return config if config is not None else {}
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Invalid YAML in {file_path}: {e}")


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two configuration dictionaries.

    Parameters
    ----------
    base : dict
        Base configuration (defaults)
    override : dict
        Override configuration (study-specific)

    Returns
    -------
    dict
        Merged configuration (override takes precedence)
    """
    merged = base.copy()

    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # Recursively merge nested dicts
            merged[key] = merge_configs(merged[key], value)
        else:
            # Override value
            merged[key] = value

    return merged


def substitute_variables(config: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Substitute environment variables and config references in strings.

    Supports:
    - ${ENV_VAR} - environment variables
    - ${config.key.subkey} - references to other config values

    Parameters
    ----------
    config : dict
        Configuration dictionary
    context : dict, optional
        Context for variable substitution (defaults to config itself)

    Returns
    -------
    dict
        Configuration with substituted values
    """
    if context is None:
        context = config

    pattern = re.compile(r'\$\{([^}]+)\}')

    def substitute_string(value: str, ctx: Dict[str, Any]) -> str:
        """Substitute variables in a single string."""
        def replacer(match):
            var_path = match.group(1)

            # Try environment variable first
            if var_path in os.environ:
                return os.environ[var_path]

            # Try config reference (e.g., ${paths.study_root})
            try:
                parts = var_path.split('.')
                val = ctx
                for part in parts:
                    val = val[part]
                return str(val)
            except (KeyError, TypeError):
                # Variable not found - leave as is
                return match.group(0)

        return pattern.sub(replacer, value)

    def process_value(value: Any, ctx: Dict[str, Any]) -> Any:
        """Recursively process values."""
        if isinstance(value, str):
            return substitute_string(value, ctx)
        elif isinstance(value, dict):
            return {k: process_value(v, ctx) for k, v in value.items()}
        elif isinstance(value, list):
            return [process_value(item, ctx) for item in value]
        else:
            return value

    # Iterate substitution to resolve chained references (e.g., A -> B -> C)
    result = config
    for _ in range(5):
        resolved = process_value(result, result)
        if resolved == result:
            break
        result = resolved

    return result


def validate_config(config: Dict[str, Any], validate_workflows: bool = False) -> None:
    """
    Validate configuration has all required parameters.

    Parameters
    ----------
    config : dict
        Configuration to validate
    validate_workflows : bool
        If True, also run per-modality workflow validation

    Raises
    ------
    ConfigurationError
        If required parameters are missing or invalid
    """
    # Check study information
    if 'study' in config:
        required_study_fields = ['name', 'code']
        for field in required_study_fields:
            if field not in config.get('study', {}):
                raise ConfigurationError(f"Missing required study field: study.{field}")

    # Check paths
    if 'paths' in config:
        required_paths = ['study_root', 'derivatives', 'transforms', 'qc']
        for path in required_paths:
            if path not in config.get('paths', {}):
                raise ConfigurationError(f"Missing required path: paths.{path}")

    # Check atlas configuration
    if 'atlas' in config:
        required_atlas_fields = ['name', 'base_path']
        for field in required_atlas_fields:
            if field not in config.get('atlas', {}):
                raise ConfigurationError(f"Missing required atlas field: atlas.{field}")

        # Validate slice definitions if present
        if 'slice_definitions' in config['atlas']:
            slice_defs = config['atlas']['slice_definitions']
            for modality, definition in slice_defs.items():
                if 'start' not in definition or 'end' not in definition:
                    raise ConfigurationError(
                        f"Atlas slice definition for '{modality}' missing 'start' or 'end'"
                    )

    # Check for at least one modality configuration
    modalities = ['anatomical', 'diffusion', 'functional', 'spectroscopy', 'msme', 'mtr']
    has_modality = any(mod in config for mod in modalities)
    if not has_modality:
        print("Warning: No modality configurations found")

    # Validate ANTs parameters if present
    if 'ants' in config:
        if 'num_threads' in config['ants']:
            num_threads = config['ants']['num_threads']
            if not isinstance(num_threads, int) or num_threads < 1:
                raise ConfigurationError(f"ants.num_threads must be positive integer, got {num_threads}")

    # Check execution parameters
    if 'execution' in config:
        if 'n_procs' in config['execution']:
            n_procs = config['execution']['n_procs']
            if not isinstance(n_procs, int) or n_procs < 1:
                raise ConfigurationError(f"execution.n_procs must be positive integer, got {n_procs}")

    print("✓ Configuration validation passed")

    if validate_workflows:
        from neurofaune.config_validator import validate_all_workflows
        import logging
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        results = validate_all_workflows(config)
        for workflow, (valid, missing_req, missing_opt) in results.items():
            if not valid:
                raise ConfigurationError(
                    f"Missing required {workflow} config keys: {missing_req}"
                )
            if missing_opt:
                print(f"  Optional {workflow} keys not set (using defaults): "
                      f"{len(missing_opt)} keys")


def default_config_path(config_path: Optional[Path] = None) -> Path:
    """The ``default.yaml`` a study config is layered on.

    A ``default.yaml`` beside the study config wins; otherwise the package's own,
    shipped inside it (``neurofaune/configs/``) so installs carry it.
    """
    if config_path is not None:
        local = Path(config_path).parent / 'default.yaml'
        if local.exists():
            return local
    return Path(__file__).parent / 'configs' / 'default.yaml'


def load_config(config_path: Path, validate: bool = True) -> Dict[str, Any]:
    """
    Load and process configuration file.

    This is the main entry point for loading configs. It:
    1. Loads the study config
    2. Merges default config < species profile (``study.species``) < study
       config; see :mod:`neurofaune.species`
    3. Substitutes variables
    4. Validates the result

    Parameters
    ----------
    config_path : Path
        Path to study-specific configuration file
    validate : bool
        Whether to validate the configuration

    Returns
    -------
    dict
        Processed configuration

    Raises
    ------
    ConfigurationError
        If configuration is invalid
    """
    config_path = Path(config_path)

    # Load study config
    study_config = load_yaml(config_path)

    default_path = default_config_path(config_path)

    from neurofaune.species import apply_species_profile

    if default_path.exists():
        default_config = load_yaml(default_path)
        # Merge: defaults < species profile < study overrides
        config = apply_species_profile(default_config, study_config)
    else:
        # No default config found - species profile + study config only
        config = apply_species_profile({}, study_config)

    # Substitute variables
    config = substitute_variables(config)

    # Validate
    if validate:
        validate_config(config)

    return config


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get a value from config using dot notation.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    key_path : str
        Dot-separated path (e.g., 'anatomical.bet.frac')
    default : any
        Default value if key not found

    Returns
    -------
    any
        Value at key_path, or default if not found

    Examples
    --------
    >>> get_config_value(config, 'anatomical.bet.frac')
    0.3
    >>> get_config_value(config, 'missing.key', default=0.3)
    0.3
    """
    try:
        parts = key_path.split('.')
        value = config
        for part in parts:
            value = value[part]
        return value
    except (KeyError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Study configs hold overrides only
# ---------------------------------------------------------------------------
#
# A study config.yaml is layered on default.yaml and the species profile when it
# is loaded (load_config). A value copied into it from default.yaml is not an
# override: it is a stale copy, and because the study config is the top layer it
# hides whatever the species profile sets for that key. init used to copy all of
# default.yaml in, so a mouse study resolved 21 of the mouse profile's 40 keys to
# rat values. Study configs therefore list only what differs from the defaults;
# slim_study_config repairs a config written the old way.

#: Sections kept whole even where they match the defaults: they say which study
#: this is, which species it is, and where it lives.
KEEP_SECTIONS = ("study", "paths")

_MISSING = object()


def _strip_matching(study: Dict[str, Any], defaults: Dict[str, Any],
                    prefix: str = "") -> tuple:
    """``study`` without the values equal to ``defaults``, and the dotted keys removed."""
    kept: Dict[str, Any] = {}
    removed = []
    for key, value in study.items():
        dotted = f"{prefix}{key}"
        default = defaults.get(key, _MISSING) if isinstance(defaults, dict) else _MISSING
        if isinstance(value, dict) and isinstance(default, dict):
            sub, sub_removed = _strip_matching(value, default, dotted + ".")
            removed += sub_removed
            if sub:
                kept[key] = sub
            elif not value:
                removed.append(dotted)
        elif value == default:
            removed.append(dotted)
        else:
            kept[key] = value
    return kept, removed


def strip_defaults(study_config: Dict[str, Any], defaults: Dict[str, Any]) -> tuple:
    """Remove every value a study config shares with ``defaults``.

    Sections in :data:`KEEP_SECTIONS` are kept whole. Returns ``(overrides,
    removed)``, where ``removed`` lists the dotted keys dropped; key order is kept.
    """
    rest = {k: v for k, v in study_config.items() if k not in KEEP_SECTIONS}
    stripped, removed = _strip_matching(rest, defaults)
    overrides = {}
    for key, value in study_config.items():
        if key in KEEP_SECTIONS:
            overrides[key] = value
        elif key in stripped:
            overrides[key] = stripped[key]
    return overrides, removed


def _flatten(config: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat = {}
    for key, value in config.items():
        if isinstance(value, dict) and value:
            flat.update(_flatten(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def resolved_differences(before: Dict[str, Any], after: Dict[str, Any],
                         defaults: Dict[str, Any]) -> list:
    """Keys whose loaded value differs between two study configs.

    Each config is resolved the way :func:`load_config` resolves it (defaults <
    species profile < study, then variable substitution). Returns
    ``[(dotted_key, value_before, value_after), ...]``; a missing key is ``None``.
    """
    from neurofaune.species import apply_species_profile

    def resolve(cfg):
        return _flatten(substitute_variables(apply_species_profile(defaults, cfg)))

    a, b = resolve(before), resolve(after)
    return [(k, a.get(k), b.get(k)) for k in sorted(set(a) | set(b))
            if a.get(k, _MISSING) != b.get(k, _MISSING)]


def slim_study_config(study_config: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce a study config to its overrides and report what that changes.

    Returns a dict with ``config`` (the slimmed config), ``removed`` (dotted keys
    dropped because they equal ``defaults``) and ``changed`` (keys whose *loaded*
    value moves, from :func:`resolved_differences`). For a rat study ``changed`` is
    empty: the rat profile is empty, so a copy of a default resolves to the same
    value. For any other species, ``changed`` lists the profile values the copied
    defaults were hiding -- the values the study was meant to have.
    """
    slim, removed = strip_defaults(study_config, defaults)
    return {
        "config": slim,
        "removed": removed,
        "changed": resolved_differences(study_config, slim, defaults),
    }


def write_study_config(config: Dict[str, Any], output_path: Path,
                       notes: Optional[list] = None) -> Path:
    """Write a study config with the header that says how it is layered."""
    header = [
        "# Neurofaune study configuration",
        "#",
        "# Lists only what this study sets. Everything else comes from the package",
        "# defaults (neurofaune/configs/default.yaml) and, above them, the species",
        "# profile named by study.species (neurofaune/configs/species/<species>.yaml).",
        "# Merge order: default.yaml < species profile < this file.",
        "#",
        "# Add a key here only to change it. A value copied from default.yaml",
        "# would hide the species profile's value for that key.",
    ]
    if notes:
        header += ["#"] + [f"# {line}" if line else "#" for line in notes]
    output_path = Path(output_path)
    with open(output_path, 'w') as f:
        f.write('\n'.join(header) + '\n\n')
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
    return output_path


def slim_config_file(config_path: Path, write: bool = False) -> Dict[str, Any]:
    """:func:`slim_study_config` on a config file, optionally rewriting it.

    With ``write=True`` the original is kept beside it as
    ``<name>.pre-slim[.N]`` and the slimmed config replaces it. Comments in the
    original are not carried over (the backup keeps them). The result dict gains
    ``backup`` (path or None).
    """
    from datetime import date

    config_path = Path(config_path)
    defaults = load_yaml(default_config_path(config_path))
    result = slim_study_config(load_yaml(config_path), defaults)
    result["backup"] = None
    if write:
        backup = config_path.with_name(config_path.name + ".pre-slim")
        n = 1
        while backup.exists():
            backup = config_path.with_name(f"{config_path.name}.pre-slim.{n}")
            n += 1
        backup.write_bytes(config_path.read_bytes())
        write_study_config(result["config"], config_path, notes=[
            f"Slimmed by `neurofaune config slim` on {date.today().isoformat()}:",
            f"{len(result['removed'])} values equal to the defaults removed;",
            f"the original is {backup.name}.",
        ])
        result["backup"] = backup
    return result
