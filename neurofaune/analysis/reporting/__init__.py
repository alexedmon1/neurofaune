"""
Unified Analysis Reporting System.

Provides a central registry for all analysis outputs and an auto-generated
HTML index dashboard. Call ``register()`` at the end of any analysis script
to record results; call ``backfill_registry()`` to discover pre-existing
summary JSONs.

Quick start::

    from neurofaune.analysis.reporting import register

    register(
        analysis_root=Path('/study/analysis'),
        entry_id='tbss_per_pnd_p60',
        analysis_type='tbss',
        display_name='TBSS: PND60 Dose Response',
        output_dir='tbss/randomise/per_pnd_p60',
        summary_stats={'n_subjects': 49, 'metrics': ['FA','MD','AD','RD']},
        source_summary_json='tbss/randomise/per_pnd_p60/analysis_summary.json',
    )
"""

from neurofaune.analysis.reporting.registry import (
    list_entries,
    load_registry,
    register,
    remove_entry,
    save_registry,
)
from neurofaune.analysis.reporting.discover import backfill_registry
from neurofaune.analysis.reporting.index_generator import generate_index_html

__all__ = [
    "register",
    "load_registry",
    "save_registry",
    "list_entries",
    "remove_entry",
    "backfill_registry",
    "generate_index_html",
]
