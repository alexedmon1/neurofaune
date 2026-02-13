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


def load_config(config_path: Path, validate: bool = True) -> Dict[str, Any]:
    """
    Load and process configuration file.

    This is the main entry point for loading configs. It:
    1. Loads the study config
    2. Loads and merges default config
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

    # Find and load default config
    # Look for default.yaml in same directory as study config
    default_path = config_path.parent / 'default.yaml'
    if not default_path.exists():
        # Try relative to package
        package_dir = Path(__file__).parent.parent
        default_path = package_dir / 'configs' / 'default.yaml'

    if default_path.exists():
        default_config = load_yaml(default_path)
        # Merge: defaults + study overrides
        config = merge_configs(default_config, study_config)
    else:
        # No default config found - use study config only
        config = study_config

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
