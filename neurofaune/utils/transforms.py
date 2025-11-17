#!/usr/bin/env python3
"""
Transformation registry for managing coordinate space transformations.

Centralizes storage and retrieval of spatial transformations (e.g., T2w→SIGMA)
to avoid duplicate computation across preprocessing workflows.

Key principle: Compute once, reuse everywhere.

Rodent-specific features:
- Support for slice-specific atlas transformations
- ANTs composite transforms (.h5) as primary format
- Metadata tracking for slice ranges used in registration
"""

from pathlib import Path
from typing import Optional, Dict, Tuple, List
import json
from datetime import datetime
import shutil


class TransformRegistry:
    """
    Registry for storing and retrieving spatial transformations.

    Manages transformations between different coordinate spaces,
    ensuring each transformation is computed only once and reused
    across all workflows that need it.

    Parameters
    ----------
    transforms_dir : Path
        Base directory for storing transformations
    subject : str
        Subject identifier (e.g., 'sub-001' or '001')
    session : str, optional
        Session identifier (for multi-session studies)
    cohort : str, optional
        Age cohort (e.g., 'p30', 'p60', 'p90')

    Examples
    --------
    >>> registry = TransformRegistry(
    ...     Path("/data/transforms"),
    ...     "sub-001",
    ...     cohort="p60"
    ... )
    >>> registry.save_ants_composite_transform(
    ...     composite_file=Path("t2w_to_sigma_Composite.h5"),
    ...     source_space="T2w",
    ...     target_space="SIGMA",
    ...     reference=Path("SIGMA_template.nii.gz")
    ... )
    >>> composite = registry.get_ants_composite_transform("T2w", "SIGMA")
    """

    def __init__(
        self,
        transforms_dir: Path,
        subject: str,
        session: Optional[str] = None,
        cohort: Optional[str] = None
    ):
        self.transforms_dir = Path(transforms_dir)
        self.subject = subject
        self.session = session
        self.cohort = cohort

        # Create subject-specific transform directory
        self.subject_dir = self._get_subject_dir()
        self.subject_dir.mkdir(parents=True, exist_ok=True)

        # Load or initialize registry metadata
        self.metadata_file = self.subject_dir / 'transforms.json'
        self.metadata = self._load_metadata()

    def _get_subject_dir(self) -> Path:
        """Get subject-specific transforms directory."""
        # Use subject ID directly without adding 'sub-' prefix if already present
        if self.subject.startswith('sub-'):
            subject_dir = self.transforms_dir / self.subject
        else:
            subject_dir = self.transforms_dir / f'sub-{self.subject}'

        if self.session:
            if not self.session.startswith('ses-'):
                session = f'ses-{self.session}'
            else:
                session = self.session
            subject_dir = subject_dir / session

        return subject_dir

    def _load_metadata(self) -> Dict:
        """Load transformation metadata from JSON file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        else:
            return {
                'subject': self.subject,
                'session': self.session,
                'cohort': self.cohort,
                'created': datetime.now().isoformat(),
                'transforms': {}
            }

    def _save_metadata(self):
        """Save transformation metadata to JSON file."""
        self.metadata['last_updated'] = datetime.now().isoformat()
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def _get_transform_key(
        self,
        source_space: str,
        target_space: str,
        modality: Optional[str] = None
    ) -> str:
        """
        Generate unique key for transformation.

        Parameters
        ----------
        source_space : str
            Source space (e.g., 'T2w', 'FA', 'b0')
        target_space : str
            Target space (e.g., 'SIGMA', 'T2w')
        modality : str, optional
            Modality identifier for slice-specific transforms (e.g., 'dwi', 'func')

        Returns
        -------
        str
            Unique transform key
        """
        if modality:
            return f"{source_space}_to_{target_space}_{modality}"
        return f"{source_space}_to_{target_space}"

    def save_ants_composite_transform(
        self,
        composite_file: Path,
        source_space: str,
        target_space: str,
        reference: Optional[Path] = None,
        source_image: Optional[Path] = None,
        modality: Optional[str] = None,
        slice_range: Optional[Tuple[int, int]] = None
    ) -> Path:
        """
        Save an ANTs composite transformation (includes all stages).

        Parameters
        ----------
        composite_file : Path
            Path to ANTs composite transform (.h5 or .mat)
        source_space : str
            Source coordinate space (e.g., 'T2w', 'FA')
        target_space : str
            Target coordinate space (e.g., 'SIGMA', 'T2w')
        reference : Path, optional
            Reference image used for transformation
        source_image : Path, optional
            Source image that was transformed
        modality : str, optional
            Modality for slice-specific transforms (e.g., 'dwi', 'func')
        slice_range : tuple, optional
            (start, end) slice range if using slice-specific atlas

        Returns
        -------
        Path
            Path to saved composite transform

        Examples
        --------
        >>> # Full atlas registration (anatomical)
        >>> registry.save_ants_composite_transform(
        ...     composite_file=Path("ants_Composite.h5"),
        ...     source_space="T2w",
        ...     target_space="SIGMA"
        ... )
        >>>
        >>> # Slice-specific registration (DTI to hippocampal slices)
        >>> registry.save_ants_composite_transform(
        ...     composite_file=Path("fa_to_sigma_Composite.h5"),
        ...     source_space="FA",
        ...     target_space="SIGMA",
        ...     modality="dwi",
        ...     slice_range=(15, 25)
        ... )
        """
        composite_file = Path(composite_file)

        if not composite_file.exists():
            raise FileNotFoundError(f"Composite transform file not found: {composite_file}")

        # Build destination path
        key = self._get_transform_key(source_space, target_space, modality)
        # Preserve original extension (.h5 or .mat)
        ext = composite_file.suffix
        dest_file = self.subject_dir / f"{key}_composite{ext}"

        # Copy file
        shutil.copy2(composite_file, dest_file)

        # Update metadata
        self.metadata['transforms'][key] = {
            'type': 'ants_composite',
            'method': 'ants',
            'source_space': source_space,
            'target_space': target_space,
            'modality': modality,
            'composite_file': str(dest_file),
            'reference': str(reference) if reference else None,
            'source_image': str(source_image) if source_image else None,
            'slice_range': slice_range,
            'created': datetime.now().isoformat()
        }

        self._save_metadata()

        return dest_file

    def save_ants_inverse_composite(
        self,
        inverse_composite_file: Path,
        source_space: str,
        target_space: str,
        reference: Optional[Path] = None,
        modality: Optional[str] = None,
        slice_range: Optional[Tuple[int, int]] = None
    ) -> Path:
        """
        Save an ANTs inverse composite transformation.

        Parameters
        ----------
        inverse_composite_file : Path
            Path to inverse composite transform
        source_space : str
            Original source space (becomes target for inverse)
        target_space : str
            Original target space (becomes source for inverse)
        reference : Path, optional
            Reference image
        modality : str, optional
            Modality identifier
        slice_range : tuple, optional
            Slice range if applicable

        Returns
        -------
        Path
            Path to saved inverse transform
        """
        inverse_composite_file = Path(inverse_composite_file)

        if not inverse_composite_file.exists():
            raise FileNotFoundError(f"Inverse composite not found: {inverse_composite_file}")

        # Build destination path (swap source/target)
        key = self._get_transform_key(target_space, source_space, modality)
        ext = inverse_composite_file.suffix
        dest_file = self.subject_dir / f"{key}_inverse_composite{ext}"

        # Copy file
        shutil.copy2(inverse_composite_file, dest_file)

        # Update metadata
        self.metadata['transforms'][key] = {
            'type': 'ants_inverse_composite',
            'method': 'ants',
            'source_space': target_space,  # Swapped
            'target_space': source_space,  # Swapped
            'modality': modality,
            'inverse_composite_file': str(dest_file),
            'reference': str(reference) if reference else None,
            'slice_range': slice_range,
            'created': datetime.now().isoformat()
        }

        self._save_metadata()

        return dest_file

    def get_ants_composite_transform(
        self,
        source_space: str,
        target_space: str,
        modality: Optional[str] = None
    ) -> Optional[Path]:
        """
        Get an ANTs composite transformation.

        Parameters
        ----------
        source_space : str
            Source coordinate space
        target_space : str
            Target coordinate space
        modality : str, optional
            Modality for slice-specific transforms

        Returns
        -------
        Path or None
            Path to composite transform, or None if not found

        Examples
        --------
        >>> # Get anatomical transform
        >>> composite = registry.get_ants_composite_transform("T2w", "SIGMA")
        >>>
        >>> # Get modality-specific transform
        >>> dwi_composite = registry.get_ants_composite_transform(
        ...     "FA", "SIGMA", modality="dwi"
        ... )
        """
        key = self._get_transform_key(source_space, target_space, modality)

        if key not in self.metadata['transforms']:
            return None

        transform_info = self.metadata['transforms'][key]

        if transform_info['type'] not in ['ants_composite', 'ants_inverse_composite']:
            print(f"Warning: Transform {key} is not an ANTs composite")
            return None

        # Get appropriate file
        if transform_info['type'] == 'ants_composite':
            composite_file = Path(transform_info['composite_file'])
        else:
            composite_file = Path(transform_info['inverse_composite_file'])

        if not composite_file.exists():
            print(f"Warning: Composite transform missing: {composite_file}")
            return None

        return composite_file

    def get_transform_metadata(
        self,
        source_space: str,
        target_space: str,
        modality: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Get metadata for a transformation (including slice range info).

        Parameters
        ----------
        source_space : str
            Source coordinate space
        target_space : str
            Target coordinate space
        modality : str, optional
            Modality identifier

        Returns
        -------
        dict or None
            Transform metadata, or None if not found
        """
        key = self._get_transform_key(source_space, target_space, modality)
        return self.metadata['transforms'].get(key)

    def has_transform(
        self,
        source_space: str,
        target_space: str,
        modality: Optional[str] = None,
        transform_type: Optional[str] = None
    ) -> bool:
        """
        Check if a transformation exists.

        Parameters
        ----------
        source_space : str
            Source coordinate space
        target_space : str
            Target coordinate space
        modality : str, optional
            Modality identifier
        transform_type : str, optional
            Type of transform ('ants_composite', etc.)
            If None, checks for any type

        Returns
        -------
        bool
            True if transformation exists

        Examples
        --------
        >>> # Check for anatomical transform
        >>> if registry.has_transform("T2w", "SIGMA"):
        ...     print("T2w→SIGMA transform available")
        >>>
        >>> # Check for modality-specific transform
        >>> if registry.has_transform("FA", "SIGMA", modality="dwi"):
        ...     print("FA→SIGMA (dwi) transform available")
        """
        key = self._get_transform_key(source_space, target_space, modality)

        if key not in self.metadata['transforms']:
            return False

        transform_info = self.metadata['transforms'][key]

        # Check type if specified
        if transform_type and transform_info['type'] != transform_type:
            return False

        # Verify files exist
        if transform_info['type'] in ['ants_composite', 'ants_inverse_composite']:
            if 'composite_file' in transform_info:
                composite_file = Path(transform_info['composite_file'])
                return composite_file.exists()
            elif 'inverse_composite_file' in transform_info:
                inverse_file = Path(transform_info['inverse_composite_file'])
                return inverse_file.exists()

        return False

    def list_transforms(self, modality: Optional[str] = None) -> List[Dict]:
        """
        List all available transformations, optionally filtered by modality.

        Parameters
        ----------
        modality : str, optional
            Filter by modality (e.g., 'dwi', 'func')

        Returns
        -------
        list
            List of transformation info dictionaries

        Examples
        --------
        >>> # List all transforms
        >>> transforms = registry.list_transforms()
        >>> for t in transforms:
        ...     print(f"{t['source_space']} → {t['target_space']} ({t['type']})")
        >>>
        >>> # List only DWI transforms
        >>> dwi_transforms = registry.list_transforms(modality='dwi')
        """
        transforms = []
        for key, info in self.metadata['transforms'].items():
            if modality is None or info.get('modality') == modality:
                transforms.append({
                    'key': key,
                    **info
                })
        return transforms

    def get_slice_range(
        self,
        source_space: str,
        target_space: str,
        modality: Optional[str] = None
    ) -> Optional[Tuple[int, int]]:
        """
        Get slice range used for a transformation (if applicable).

        Parameters
        ----------
        source_space : str
            Source coordinate space
        target_space : str
            Target coordinate space
        modality : str, optional
            Modality identifier

        Returns
        -------
        tuple or None
            (start, end) slice range, or None if not slice-specific

        Examples
        --------
        >>> slice_range = registry.get_slice_range("FA", "SIGMA", modality="dwi")
        >>> if slice_range:
        ...     start, end = slice_range
        ...     print(f"Transform uses slices {start} to {end}")
        """
        metadata = self.get_transform_metadata(source_space, target_space, modality)
        if metadata:
            return metadata.get('slice_range')
        return None

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate that all registered transformations exist on disk.

        Returns
        -------
        tuple
            (is_valid, missing_files) - validation result and list of missing files

        Examples
        --------
        >>> is_valid, missing = registry.validate()
        >>> if not is_valid:
        ...     print(f"Missing files: {missing}")
        """
        missing_files = []

        for key, info in self.metadata['transforms'].items():
            if info['type'] in ['ants_composite', 'ants_inverse_composite']:
                if 'composite_file' in info:
                    file_path = Path(info['composite_file'])
                elif 'inverse_composite_file' in info:
                    file_path = Path(info['inverse_composite_file'])
                else:
                    missing_files.append(f"{key}: No file path found")
                    continue

                if not file_path.exists():
                    missing_files.append(str(file_path))

        return len(missing_files) == 0, missing_files

    def get_metadata(self) -> Dict:
        """
        Get full registry metadata.

        Returns
        -------
        dict
            Complete metadata dictionary
        """
        return self.metadata.copy()


def create_transform_registry(
    config: Dict,
    subject: str,
    session: Optional[str] = None,
    cohort: Optional[str] = None
) -> TransformRegistry:
    """
    Create a TransformRegistry from configuration.

    Parameters
    ----------
    config : dict
        Configuration dictionary with 'paths.transforms' key
    subject : str
        Subject identifier
    session : str, optional
        Session identifier
    cohort : str, optional
        Age cohort (e.g., 'p30', 'p60', 'p90')

    Returns
    -------
    TransformRegistry
        Initialized registry

    Examples
    --------
    >>> from neurofaune.config import load_config
    >>> config = load_config("study.yaml")
    >>> registry = create_transform_registry(config, "sub-001", cohort="p60")
    """
    transforms_dir = Path(config['paths']['transforms'])
    return TransformRegistry(transforms_dir, subject, session, cohort)
