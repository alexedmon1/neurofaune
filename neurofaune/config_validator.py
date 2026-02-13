"""
Configuration Validators for MRI Preprocessing Workflows.

This module provides validation functions for each modality workflow,
ensuring all required parameters are present before execution.
Ported from neurovrai's config_validator pattern with rodent-specific keys.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


class ConfigValidator:
    """Base configuration validator."""

    @staticmethod
    def check_required_keys(
        config: Dict,
        required_keys: List[str],
        section: str = ""
    ) -> Tuple[bool, List[str]]:
        """
        Check if required keys exist in config.

        Parameters
        ----------
        config : dict
            Configuration dictionary
        required_keys : list
            List of required key paths (e.g., 'anatomical.skull_strip.method')
        section : str
            Section name for logging

        Returns
        -------
        Tuple[bool, List[str]]
            (is_valid, missing_keys)
        """
        missing_keys = []

        for key_path in required_keys:
            keys = key_path.split('.')
            current = config

            for key in keys:
                if not isinstance(current, dict) or key not in current:
                    missing_keys.append(key_path)
                    break
                current = current[key]

        is_valid = len(missing_keys) == 0
        return is_valid, missing_keys

    @staticmethod
    def check_file_exists(file_path: Optional[Path], param_name: str) -> bool:
        """
        Check if a file exists.

        Parameters
        ----------
        file_path : Path or None
            Path to check
        param_name : str
            Parameter name for logging

        Returns
        -------
        bool
            True if exists or None, False otherwise
        """
        if file_path is None:
            return True

        if not Path(file_path).exists():
            logger.warning(f"{param_name} not found: {file_path}")
            return False

        return True


class AnatomicalConfigValidator(ConfigValidator):
    """Validator for anatomical preprocessing workflow."""

    REQUIRED_KEYS = [
        'anatomical.skull_strip.method',
    ]

    OPTIONAL_KEYS = [
        'anatomical.skull_strip.n_classes',
        'anatomical.skull_strip.atropos_iterations',
        'anatomical.skull_strip.atropos_convergence',
        'anatomical.skull_strip.mrf_smoothing_factor',
        'anatomical.skull_strip.mrf_radius',
        'anatomical.skull_strip.tissue_confidence_threshold',
        'anatomical.skull_strip.morphological_closing_iterations',
        'anatomical.skull_strip.foreground_otsu_multiplier',
        'anatomical.skull_strip.adaptive_bet.cnr_thresholds',
        'anatomical.skull_strip.adaptive_bet.frac_mapping',
        'anatomical.n4.iterations',
        'anatomical.n4.shrink_factor',
        'anatomical.n4.convergence_threshold',
        'anatomical.intensity_normalization.factor',
        'anatomical.registration.smoothing_sigmas',
        'anatomical.registration.shrink_factors',
        'anatomical.registration.iterations',
        'anatomical.registration.convergence_threshold',
        'anatomical.registration.convergence_window_size',
        'anatomical.registration.syn_params',
        'anatomical.registration.metric_bins',
    ]

    @classmethod
    def validate(cls, config: Dict) -> Tuple[bool, List[str], List[str]]:
        """
        Validate anatomical preprocessing configuration.

        Returns
        -------
        Tuple[bool, List[str], List[str]]
            (is_valid, missing_required, missing_optional)
        """
        logger.info("Validating anatomical preprocessing configuration...")

        is_valid, missing_required = cls.check_required_keys(
            config, cls.REQUIRED_KEYS, 'anatomical'
        )

        _, missing_optional = cls.check_required_keys(
            config, cls.OPTIONAL_KEYS, 'anatomical'
        )

        if is_valid:
            logger.info("  ✓ Anatomical config valid")
        else:
            logger.error("  ✗ Anatomical config invalid")
            for key in missing_required:
                logger.error(f"    Missing required: {key}")

        if missing_optional:
            logger.warning("  ⚠ Missing optional parameters (will use defaults):")
            for key in missing_optional:
                logger.warning(f"    {key}")

        return is_valid, missing_required, missing_optional


class DWIConfigValidator(ConfigValidator):
    """Validator for DWI preprocessing workflow."""

    REQUIRED_KEYS = [
        'diffusion.eddy.repol',
    ]

    OPTIONAL_KEYS = [
        'diffusion.eddy.phase_encoding_direction',
        'diffusion.eddy.readout_time',
        'diffusion.eddy.data_is_shelled',
        'diffusion.brain_extraction.method',
        'diffusion.intensity_normalization.target_max',
        'diffusion.eddy.slice_padding',
        'diffusion.topup.readout_time',
    ]

    @classmethod
    def validate(cls, config: Dict) -> Tuple[bool, List[str], List[str]]:
        """
        Validate DWI preprocessing configuration.

        Returns
        -------
        Tuple[bool, List[str], List[str]]
            (is_valid, missing_required, missing_optional)
        """
        logger.info("Validating DWI preprocessing configuration...")

        is_valid, missing_required = cls.check_required_keys(
            config, cls.REQUIRED_KEYS, 'diffusion'
        )

        _, missing_optional = cls.check_required_keys(
            config, cls.OPTIONAL_KEYS, 'diffusion'
        )

        if is_valid:
            logger.info("  ✓ DWI config valid")
        else:
            logger.error("  ✗ DWI config invalid")
            for key in missing_required:
                logger.error(f"    Missing required: {key}")

        if missing_optional:
            logger.warning("  ⚠ Missing optional parameters (will use defaults):")
            for key in missing_optional:
                logger.warning(f"    {key}")

        return is_valid, missing_required, missing_optional


class FunctionalConfigValidator(ConfigValidator):
    """Validator for functional MRI preprocessing workflow."""

    REQUIRED_KEYS = [
        'functional.tr',
    ]

    OPTIONAL_KEYS = [
        'functional.skull_strip_adaptive.target_ratio',
        'functional.skull_strip_adaptive.frac_range',
        'functional.skull_strip_adaptive.frac_step',
        'functional.denoising.ica.motion_threshold',
        'functional.denoising.ica.edge_threshold',
        'functional.denoising.ica.csf_threshold',
        'functional.denoising.ica.freq_threshold',
        'functional.motion_qc.fd_threshold',
    ]

    @classmethod
    def validate(cls, config: Dict) -> Tuple[bool, List[str], List[str]]:
        """
        Validate functional preprocessing configuration.

        Returns
        -------
        Tuple[bool, List[str], List[str]]
            (is_valid, missing_required, missing_optional)
        """
        logger.info("Validating functional preprocessing configuration...")

        is_valid, missing_required = cls.check_required_keys(
            config, cls.REQUIRED_KEYS, 'functional'
        )

        _, missing_optional = cls.check_required_keys(
            config, cls.OPTIONAL_KEYS, 'functional'
        )

        if is_valid:
            logger.info("  ✓ Functional config valid")
        else:
            logger.error("  ✗ Functional config invalid")
            for key in missing_required:
                logger.error(f"    Missing required: {key}")

        if missing_optional:
            logger.warning("  ⚠ Missing optional parameters (will use defaults):")
            for key in missing_optional:
                logger.warning(f"    {key}")

        return is_valid, missing_required, missing_optional


class MSMEConfigValidator(ConfigValidator):
    """Validator for MSME T2 mapping preprocessing workflow."""

    REQUIRED_KEYS = []

    OPTIONAL_KEYS = [
        'msme.skull_strip.method',
        'msme.skull_strip.n_classes',
        'msme.t2_fitting.n_components',
        'msme.t2_fitting.t2_range',
        'msme.t2_fitting.lambda_reg',
        'msme.t2_fitting.myelin_water_cutoff',
        'msme.t2_fitting.intra_extra_cutoff',
    ]

    @classmethod
    def validate(cls, config: Dict) -> Tuple[bool, List[str], List[str]]:
        """
        Validate MSME preprocessing configuration.

        Returns
        -------
        Tuple[bool, List[str], List[str]]
            (is_valid, missing_required, missing_optional)
        """
        logger.info("Validating MSME preprocessing configuration...")

        is_valid, missing_required = cls.check_required_keys(
            config, cls.REQUIRED_KEYS, 'msme'
        )

        _, missing_optional = cls.check_required_keys(
            config, cls.OPTIONAL_KEYS, 'msme'
        )

        if is_valid:
            logger.info("  ✓ MSME config valid")
        else:
            logger.error("  ✗ MSME config invalid")
            for key in missing_required:
                logger.error(f"    Missing required: {key}")

        if missing_optional:
            logger.warning("  ⚠ Missing optional parameters (will use defaults):")
            for key in missing_optional:
                logger.warning(f"    {key}")

        return is_valid, missing_required, missing_optional


def validate_all_workflows(config: Dict) -> Dict[str, Tuple[bool, List[str], List[str]]]:
    """
    Validate configuration for all workflows.

    Parameters
    ----------
    config : dict
        Configuration dictionary

    Returns
    -------
    dict
        Dictionary mapping workflow name to (is_valid, missing_required, missing_optional)
    """
    logger.info("=" * 70)
    logger.info("VALIDATING CONFIGURATION FOR ALL WORKFLOWS")
    logger.info("=" * 70)
    logger.info("")

    validators = {
        'anatomical': AnatomicalConfigValidator,
        'dwi': DWIConfigValidator,
        'functional': FunctionalConfigValidator,
        'msme': MSMEConfigValidator,
    }

    results = {}
    for workflow, validator_class in validators.items():
        is_valid, missing_req, missing_opt = validator_class.validate(config)
        results[workflow] = (is_valid, missing_req, missing_opt)
        logger.info("")

    # Summary
    logger.info("=" * 70)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 70)

    all_valid = all(r[0] for r in results.values())
    for workflow, (is_valid, _, _) in results.items():
        status = "✓" if is_valid else "✗"
        logger.info(f"  {status} {workflow}: {'VALID' if is_valid else 'INVALID'}")

    logger.info("")

    if all_valid:
        logger.info("All workflows validated successfully")
    else:
        logger.warning("Some workflows have invalid configurations")

    return results
