"""
Template manifest for tracking template building metadata.

This module provides dataclasses and utilities for tracking which subjects
were used to build templates, preprocessing parameters, and transform locations.
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import nibabel as nib
import numpy as np


@dataclass
class SubjectTransforms:
    """Track transforms for a single subject used in template building."""
    subject: str
    session: str
    preprocessed_t2w: Path
    brain_mask: Path
    to_template_affine: Optional[Path] = None
    to_template_warp: Optional[Path] = None
    to_template_inverse_warp: Optional[Path] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with string paths."""
        return {
            'subject': self.subject,
            'session': self.session,
            'preprocessed_t2w': str(self.preprocessed_t2w),
            'brain_mask': str(self.brain_mask),
            'to_template_affine': str(self.to_template_affine) if self.to_template_affine else None,
            'to_template_warp': str(self.to_template_warp) if self.to_template_warp else None,
            'to_template_inverse_warp': str(self.to_template_inverse_warp) if self.to_template_inverse_warp else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SubjectTransforms':
        """Create from dictionary."""
        return cls(
            subject=data['subject'],
            session=data['session'],
            preprocessed_t2w=Path(data['preprocessed_t2w']),
            brain_mask=Path(data['brain_mask']),
            to_template_affine=Path(data['to_template_affine']) if data.get('to_template_affine') else None,
            to_template_warp=Path(data['to_template_warp']) if data.get('to_template_warp') else None,
            to_template_inverse_warp=Path(data['to_template_inverse_warp']) if data.get('to_template_inverse_warp') else None,
        )


@dataclass
class TemplateManifest:
    """
    Comprehensive manifest for template building.

    Tracks:
    - Which subjects were used
    - Preprocessing parameters
    - Transform locations
    - QC metrics
    - Timestamps
    """
    # Basic info
    study_name: str
    cohort: str
    modality: str

    # Template info
    template_path: Path
    n_subjects: int
    fraction_used: float
    selection_method: str  # 'random', 'quality', or 'manual'

    # Subjects used
    subjects_used: List[str] = field(default_factory=list)
    sessions_used: List[str] = field(default_factory=list)
    subject_transforms: Dict[str, SubjectTransforms] = field(default_factory=dict)

    # SIGMA registration
    sigma_affine: Optional[Path] = None
    sigma_warp: Optional[Path] = None
    sigma_inverse_warp: Optional[Path] = None
    sigma_warped_template: Optional[Path] = None

    # QC metrics
    qc_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    template_sigma_correlation: Optional[float] = None
    template_sigma_dice: Optional[float] = None

    # Preprocessing parameters
    preprocessing_params: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    neurofaune_version: str = "0.1.0"

    def save(self, output_path: Path) -> Path:
        """Save manifest to JSON file."""
        data = {
            'study_name': self.study_name,
            'cohort': self.cohort,
            'modality': self.modality,
            'template_path': str(self.template_path),
            'n_subjects': self.n_subjects,
            'fraction_used': self.fraction_used,
            'selection_method': self.selection_method,
            'subjects_used': self.subjects_used,
            'sessions_used': self.sessions_used,
            'subject_transforms': {k: v.to_dict() for k, v in self.subject_transforms.items()},
            'sigma_affine': str(self.sigma_affine) if self.sigma_affine else None,
            'sigma_warp': str(self.sigma_warp) if self.sigma_warp else None,
            'sigma_inverse_warp': str(self.sigma_inverse_warp) if self.sigma_inverse_warp else None,
            'sigma_warped_template': str(self.sigma_warped_template) if self.sigma_warped_template else None,
            'qc_metrics': self.qc_metrics,
            'template_sigma_correlation': self.template_sigma_correlation,
            'template_sigma_dice': self.template_sigma_dice,
            'preprocessing_params': self.preprocessing_params,
            'created_at': self.created_at,
            'neurofaune_version': self.neurofaune_version,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Template manifest saved to: {output_path}")
        return output_path

    @classmethod
    def load(cls, manifest_path: Path) -> 'TemplateManifest':
        """Load manifest from JSON file."""
        with open(manifest_path, 'r') as f:
            data = json.load(f)

        # Convert paths
        subject_transforms = {}
        for k, v in data.get('subject_transforms', {}).items():
            subject_transforms[k] = SubjectTransforms.from_dict(v)

        return cls(
            study_name=data['study_name'],
            cohort=data['cohort'],
            modality=data['modality'],
            template_path=Path(data['template_path']),
            n_subjects=data['n_subjects'],
            fraction_used=data['fraction_used'],
            selection_method=data['selection_method'],
            subjects_used=data['subjects_used'],
            sessions_used=data['sessions_used'],
            subject_transforms=subject_transforms,
            sigma_affine=Path(data['sigma_affine']) if data.get('sigma_affine') else None,
            sigma_warp=Path(data['sigma_warp']) if data.get('sigma_warp') else None,
            sigma_inverse_warp=Path(data['sigma_inverse_warp']) if data.get('sigma_inverse_warp') else None,
            sigma_warped_template=Path(data['sigma_warped_template']) if data.get('sigma_warped_template') else None,
            qc_metrics=data.get('qc_metrics', {}),
            template_sigma_correlation=data.get('template_sigma_correlation'),
            template_sigma_dice=data.get('template_sigma_dice'),
            preprocessing_params=data.get('preprocessing_params', {}),
            created_at=data.get('created_at', ''),
            neurofaune_version=data.get('neurofaune_version', ''),
        )

    def is_template_subject(self, subject: str, session: str) -> bool:
        """Check if a subject/session was used in template building."""
        key = f"{subject}_{session}"
        return key in self.subject_transforms

    def get_subject_transforms(self, subject: str, session: str) -> Optional[SubjectTransforms]:
        """Get transforms for a template subject."""
        key = f"{subject}_{session}"
        return self.subject_transforms.get(key)

    def add_subject(
        self,
        subject: str,
        session: str,
        preprocessed_t2w: Path,
        brain_mask: Path,
        qc_metrics: Optional[Dict[str, float]] = None
    ):
        """Add a subject to the manifest."""
        key = f"{subject}_{session}"
        self.subject_transforms[key] = SubjectTransforms(
            subject=subject,
            session=session,
            preprocessed_t2w=preprocessed_t2w,
            brain_mask=brain_mask
        )

        if subject not in self.subjects_used:
            self.subjects_used.append(subject)
        if session not in self.sessions_used:
            self.sessions_used.append(session)

        if qc_metrics:
            self.qc_metrics[key] = qc_metrics

        self.n_subjects = len(self.subject_transforms)


def find_template_manifest(
    templates_dir: Path,
    cohort: str,
    modality: str = 'anat'
) -> Optional[Path]:
    """
    Find template manifest for a given cohort.

    Parameters
    ----------
    templates_dir : Path
        Root templates directory
    cohort : str
        Age cohort ('p30', 'p60', 'p90')
    modality : str
        Modality (default: 'anat')

    Returns
    -------
    Optional[Path]
        Path to manifest file if found, None otherwise
    """
    manifest_path = templates_dir / modality / cohort / 'template_manifest.json'
    if manifest_path.exists():
        return manifest_path

    # Try alternative locations
    alt_path = templates_dir / modality / cohort / f'tpl-manifest_{cohort}_{modality}.json'
    if alt_path.exists():
        return alt_path

    return None
