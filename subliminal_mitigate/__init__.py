"""Reusable components for the subliminal-mitigation research pipeline.

Historical experiment scripts remain at their original paths because several
sealed workflows bind those files by path and SHA-256.  New reusable code
belongs in this package; experiment-specific scripts should be thin drivers.
"""

from .pipeline import REQUIRED_STAGE_NAMES, ResearchPipeline

__all__ = ["REQUIRED_STAGE_NAMES", "ResearchPipeline"]
__version__ = "0.1.0"
