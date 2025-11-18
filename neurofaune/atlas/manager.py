#!/usr/bin/env python3
"""
SIGMA Atlas Manager.

Provides unified interface for accessing SIGMA rat brain atlas components:
- Brain templates (InVivo/ExVivo)
- Tissue probability maps (GM, WM, CSF)
- Parcellation atlases (anatomical and functional)
- ROI labels and metadata
- Modality-specific slice extraction
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import nibabel as nib
import numpy as np
import pandas as pd

from neurofaune.atlas.slice_extraction import extract_slices

logger = logging.getLogger(__name__)


class AtlasManager:
    """
    Manager for SIGMA rat brain atlas.

    Provides access to templates, tissue masks, parcellations, and ROI labels.
    Supports slice extraction for modality-specific registration.

    Parameters
    ----------
    config : dict
        Configuration dictionary with atlas settings
    atlas_type : str, default='invivo'
        Atlas type: 'invivo' or 'exvivo'

    Attributes
    ----------
    base_path : Path
        Base directory containing SIGMA atlas files
    atlas_type : str
        Atlas type (invivo or exvivo)
    templates : dict
        Cache of loaded template images
    labels : dict
        Cache of loaded label images
    roi_definitions : DataFrame
        ROI metadata from CSV file

    Examples
    --------
    >>> from neurofaune.config import load_config
    >>> from neurofaune.atlas import AtlasManager
    >>>
    >>> config = load_config('config.yaml')
    >>> atlas = AtlasManager(config)
    >>>
    >>> # Get full brain template
    >>> template = atlas.get_template()
    >>>
    >>> # Get template with hippocampal slices only (for DTI)
    >>> dwi_template = atlas.get_template(modality='dwi')
    >>>
    >>> # Get tissue masks
    >>> gm_mask = atlas.get_tissue_mask('gm')
    >>>
    >>> # Get parcellation with ROI labels
    >>> parcellation = atlas.get_parcellation()
    >>> roi_labels = atlas.get_roi_labels()
    """

    def __init__(self, config: Dict[str, Any], atlas_type: str = 'invivo'):
        """Initialize atlas manager."""
        self.config = config
        self.atlas_type = atlas_type.lower()

        if self.atlas_type not in ['invivo', 'exvivo']:
            raise ValueError(f"atlas_type must be 'invivo' or 'exvivo', got '{atlas_type}'")

        # Get atlas path from config
        if 'atlas' not in config or 'base_path' not in config['atlas']:
            raise ValueError("Configuration must contain 'atlas.base_path'")

        self.base_path = Path(config['atlas']['base_path'])

        if not self.base_path.exists():
            raise FileNotFoundError(f"Atlas directory not found: {self.base_path}")

        # Define atlas subdirectories
        self._anatomical_dir = self.base_path / 'SIGMA_Rat_Anatomical_Imaging'
        self._atlas_dir = self.base_path / 'SIGMA_Rat_Brain_Atlases'
        self._functional_dir = self.base_path / 'SIGMA_Rat_Functional_Imaging'

        # Cache for loaded images (avoid re-reading)
        self.templates = {}
        self.labels = {}

        # Load ROI definitions
        self.roi_definitions = self._load_roi_definitions()

        logger.info(f"Initialized SIGMA AtlasManager ({atlas_type}) from {self.base_path}")

    def _load_roi_definitions(self) -> pd.DataFrame:
        """
        Load ROI definitions from CSV file.

        Returns
        -------
        DataFrame
            ROI metadata with columns: Labels, Hemisphere, Matter, Territories,
            System, Region of interest
        """
        csv_path = self.base_path / 'SIGMA_InVivo_Anatomical_Brain_Atlas_Labels.csv'

        if not csv_path.exists():
            logger.warning(f"ROI definitions not found: {csv_path}")
            return pd.DataFrame()

        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(df)} ROI definitions from {csv_path.name}")
            return df
        except Exception as e:
            logger.error(f"Failed to load ROI definitions: {e}")
            return pd.DataFrame()

    def get_template(
        self,
        modality: Optional[str] = None,
        masked: bool = True,
        coronal: bool = False
    ) -> nib.Nifti1Image:
        """
        Get brain template image.

        Parameters
        ----------
        modality : str, optional
            Modality for slice extraction ('dwi', 'func', 'anat', etc.)
            If None, returns full brain template
        masked : bool, default=True
            Return skull-stripped template
        coronal : bool, default=False
            Return coronal-oriented template (for registration)

        Returns
        -------
        Nifti1Image
            Brain template image (optionally slice-extracted)

        Examples
        --------
        >>> # Full brain template
        >>> template = atlas.get_template()
        >>>
        >>> # Hippocampal slices for DTI
        >>> dwi_template = atlas.get_template(modality='dwi')
        >>>
        >>> # Coronal orientation for registration
        >>> reg_template = atlas.get_template(coronal=True)
        """
        # Build cache key
        cache_key = f"{self.atlas_type}_{'masked' if masked else 'full'}_{'coronal' if coronal else 'axial'}_{modality or 'all'}"

        # Check cache
        if cache_key in self.templates:
            logger.debug(f"Using cached template: {cache_key}")
            return self.templates[cache_key]

        # Determine template path
        if self.atlas_type == 'invivo':
            template_dir = self._anatomical_dir / 'SIGMA_Rat_Anatomical_InVivo_Template'

            if modality == 'func':
                # Use EPI template for functional data
                template_path = self._functional_dir / 'SIGMA_EPI_Brain_Template_Masked_Coronal.nii.gz'
            elif modality == 'dwi' and masked:
                # Use hippocampus-focused template for DTI
                template_path = template_dir / 'SIGMA_InVivo_Brain_Template_Masked_Coronal_Hippocampus.nii.gz'
            else:
                # Standard anatomical template
                if masked and coronal:
                    template_path = template_dir / 'SIGMA_InVivo_Brain_Template_Masked_Coronal.nii'
                elif masked:
                    template_path = template_dir / 'SIGMA_InVivo_Brain_Template_Masked.nii'
                else:
                    template_path = template_dir / 'SIGMA_InVivo_Brain_Template.nii'
        else:
            # ExVivo template
            template_dir = self._anatomical_dir / 'SIGMA_Rat_Anatomical_ExVivo_Template'
            if masked:
                template_path = template_dir / 'SIGMA_ExVivo_Brain_Template_Masked.nii'
            else:
                template_path = template_dir / 'SIGMA_ExVivo_Brain_Template.nii'

        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")

        # Load template
        logger.info(f"Loading template: {template_path.name}")
        template_img = nib.load(template_path)

        # Apply slice extraction if modality specified
        if modality and 'slice_definitions' in self.config.get('atlas', {}):
            slice_def = self.config['atlas']['slice_definitions'].get(modality)
            if slice_def:
                logger.info(f"Extracting slices for {modality}: {slice_def['start']} to {slice_def['end']}")
                template_img = extract_slices(
                    template_img,
                    slice_start=slice_def['start'],
                    slice_end=slice_def['end']
                )

        # Cache and return
        self.templates[cache_key] = template_img
        return template_img

    def get_tissue_mask(
        self,
        tissue_type: str,
        modality: Optional[str] = None,
        probability: bool = False
    ) -> nib.Nifti1Image:
        """
        Get tissue probability map or binary mask.

        Parameters
        ----------
        tissue_type : str
            Tissue type: 'gm', 'wm', or 'csf'
        modality : str, optional
            Modality for slice extraction
        probability : bool, default=False
            Return probability map (if False, returns binary mask)

        Returns
        -------
        Nifti1Image
            Tissue mask or probability map

        Examples
        --------
        >>> # Binary GM mask
        >>> gm_mask = atlas.get_tissue_mask('gm')
        >>>
        >>> # GM probability map
        >>> gm_prob = atlas.get_tissue_mask('gm', probability=True)
        >>>
        >>> # WM mask for DTI slices
        >>> wm_dwi = atlas.get_tissue_mask('wm', modality='dwi')
        """
        tissue_type = tissue_type.lower()
        if tissue_type not in ['gm', 'wm', 'csf']:
            raise ValueError(f"tissue_type must be 'gm', 'wm', or 'csf', got '{tissue_type}'")

        # Build cache key
        cache_key = f"{self.atlas_type}_{tissue_type}_{'prob' if probability else 'mask'}_{modality or 'all'}"

        # Check cache
        if cache_key in self.templates:
            return self.templates[cache_key]

        # Determine file path
        if self.atlas_type == 'invivo':
            tissue_dir = self._anatomical_dir / 'SIGMA_Rat_Anatomical_InVivo_Template'
            tissue_map = {'gm': 'GM', 'wm': 'WM', 'csf': 'CSF'}
            tissue_name = tissue_map[tissue_type]

            if probability:
                # Probability map
                tissue_path = tissue_dir / f'SIGMA_InVivo_{tissue_name}.nii'
            else:
                # Binary mask
                tissue_path = tissue_dir / f'SIGMA_InVivo_{tissue_name}_mask.nii'
        else:
            # ExVivo
            tissue_dir = self._anatomical_dir / 'SIGMA_Rat_Anatomical_ExVivo_Template'
            tissue_map = {'gm': 'GM', 'wm': 'WM', 'csf': 'CSF'}
            tissue_name = tissue_map[tissue_type]
            tissue_path = tissue_dir / f'SIGMA_ExVivo_{tissue_name}.nii'

        if not tissue_path.exists():
            raise FileNotFoundError(f"Tissue mask not found: {tissue_path}")

        # Load tissue mask
        logger.info(f"Loading tissue mask: {tissue_path.name}")
        tissue_img = nib.load(tissue_path)

        # Apply slice extraction if needed
        if modality and 'slice_definitions' in self.config.get('atlas', {}):
            slice_def = self.config['atlas']['slice_definitions'].get(modality)
            if slice_def:
                tissue_img = extract_slices(
                    tissue_img,
                    slice_start=slice_def['start'],
                    slice_end=slice_def['end']
                )

        # Cache and return
        self.templates[cache_key] = tissue_img
        return tissue_img

    def get_brain_mask(
        self,
        modality: Optional[str] = None
    ) -> nib.Nifti1Image:
        """
        Get binary brain mask.

        Parameters
        ----------
        modality : str, optional
            Modality for slice extraction

        Returns
        -------
        Nifti1Image
            Binary brain mask
        """
        cache_key = f"{self.atlas_type}_brain_mask_{modality or 'all'}"

        if cache_key in self.templates:
            return self.templates[cache_key]

        if self.atlas_type == 'invivo':
            mask_dir = self._anatomical_dir / 'SIGMA_Rat_Anatomical_InVivo_Template'
            mask_path = mask_dir / 'SIGMA_InVivo_Brain_Mask.nii'
        else:
            mask_dir = self._anatomical_dir / 'SIGMA_Rat_Anatomical_ExVivo_Template'
            mask_path = mask_dir / 'SIGMA_ExVivo_Brain_Mask.nii'

        if not mask_path.exists():
            raise FileNotFoundError(f"Brain mask not found: {mask_path}")

        logger.info(f"Loading brain mask: {mask_path.name}")
        mask_img = nib.load(mask_path)

        # Apply slice extraction if needed
        if modality and 'slice_definitions' in self.config.get('atlas', {}):
            slice_def = self.config['atlas']['slice_definitions'].get(modality)
            if slice_def:
                mask_img = extract_slices(
                    mask_img,
                    slice_start=slice_def['start'],
                    slice_end=slice_def['end']
                )

        self.templates[cache_key] = mask_img
        return mask_img

    def get_parcellation(
        self,
        atlas_type: str = 'anatomical',
        modality: Optional[str] = None
    ) -> nib.Nifti1Image:
        """
        Get parcellation atlas with ROI labels.

        Parameters
        ----------
        atlas_type : str, default='anatomical'
            Atlas type: 'anatomical' or 'functional'
        modality : str, optional
            Modality for slice extraction

        Returns
        -------
        Nifti1Image
            Parcellation image with integer ROI labels

        Examples
        --------
        >>> # Get anatomical parcellation
        >>> parcellation = atlas.get_parcellation('anatomical')
        >>>
        >>> # Get functional parcellation for fMRI
        >>> func_parcellation = atlas.get_parcellation('functional', modality='func')
        """
        cache_key = f"parcellation_{atlas_type}_{self.atlas_type}_{modality or 'all'}"

        if cache_key in self.labels:
            return self.labels[cache_key]

        if atlas_type == 'anatomical':
            if self.atlas_type == 'invivo':
                parc_path = (self._atlas_dir / 'SIGMA_Anatomical_Atlas' / 'InVivo_Atlas' /
                            'SIGMA_InVivo_Anatomical_Brain_Atlas.nii')
            else:
                parc_path = (self._atlas_dir / 'SIGMA_Anatomical_Atlas' / 'ExVivo_Atlas' /
                            'SIGMA_ExVivo_Anatomical_Brain_Atlas.nii')
        elif atlas_type == 'functional':
            parc_path = (self._atlas_dir / 'SIGMA_Functional_Atlas' /
                        'SIGMA_Functional_Brain_Atlas_InVivo_Anatomical_Template.nii')
        else:
            raise ValueError(f"atlas_type must be 'anatomical' or 'functional', got '{atlas_type}'")

        if not parc_path.exists():
            raise FileNotFoundError(f"Parcellation not found: {parc_path}")

        logger.info(f"Loading parcellation: {parc_path.name}")
        parc_img = nib.load(parc_path)

        # Apply slice extraction if needed
        if modality and 'slice_definitions' in self.config.get('atlas', {}):
            slice_def = self.config['atlas']['slice_definitions'].get(modality)
            if slice_def:
                parc_img = extract_slices(
                    parc_img,
                    slice_start=slice_def['start'],
                    slice_end=slice_def['end']
                )

        self.labels[cache_key] = parc_img
        return parc_img

    def get_roi_labels(self) -> pd.DataFrame:
        """
        Get ROI label definitions.

        Returns
        -------
        DataFrame
            ROI metadata with columns:
            - Labels: Integer label value
            - Hemisphere: L/R/M (left/right/midline)
            - Matter: Grey Matter / White Matter
            - Territories: Anatomical territory
            - System: Brain system
            - Region of interest: Full ROI name

        Examples
        --------
        >>> roi_df = atlas.get_roi_labels()
        >>> hippocampus = roi_df[roi_df['Region of interest'].str.contains('Hippocampus')]
        """
        return self.roi_definitions

    def get_roi_mask(
        self,
        roi_name: str,
        modality: Optional[str] = None,
        atlas_type: str = 'anatomical'
    ) -> nib.Nifti1Image:
        """
        Create binary mask for a specific ROI.

        Parameters
        ----------
        roi_name : str
            ROI name (partial match allowed)
        modality : str, optional
            Modality for slice extraction
        atlas_type : str, default='anatomical'
            Parcellation to use

        Returns
        -------
        Nifti1Image
            Binary mask for specified ROI

        Examples
        --------
        >>> # Get hippocampus mask
        >>> hipp_mask = atlas.get_roi_mask('Hippocampus')
        >>>
        >>> # Get CA1 region
        >>> ca1_mask = atlas.get_roi_mask('Cornu.Ammonis.1')
        """
        # Find matching ROIs
        matches = self.roi_definitions[
            self.roi_definitions['Region of interest'].str.contains(roi_name, case=False, na=False)
        ]

        if len(matches) == 0:
            raise ValueError(f"No ROIs found matching '{roi_name}'")

        logger.info(f"Found {len(matches)} ROIs matching '{roi_name}'")

        # Get parcellation
        parcellation = self.get_parcellation(atlas_type=atlas_type, modality=modality)

        # Create binary mask
        parc_data = parcellation.get_fdata()
        mask_data = np.isin(parc_data, matches['Labels'].values).astype(np.uint8)

        # Create new image with same header
        mask_img = nib.Nifti1Image(mask_data, parcellation.affine, parcellation.header)

        return mask_img

    def get_slice_range(self, modality: str) -> Tuple[int, int]:
        """
        Get configured slice range for a modality.

        Parameters
        ----------
        modality : str
            Modality name

        Returns
        -------
        tuple of int
            (start_slice, end_slice)

        Raises
        ------
        ValueError
            If modality not found in slice definitions
        """
        if 'slice_definitions' not in self.config.get('atlas', {}):
            raise ValueError("No slice_definitions in atlas config")

        slice_defs = self.config['atlas']['slice_definitions']
        if modality not in slice_defs:
            raise ValueError(f"No slice definition for modality '{modality}'")

        slice_def = slice_defs[modality]
        return (slice_def['start'], slice_def['end'])

    def clear_cache(self):
        """Clear cached images to free memory."""
        self.templates.clear()
        self.labels.clear()
        logger.info("Cleared atlas image cache")
