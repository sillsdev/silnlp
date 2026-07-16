"""Combined NLLB web demo.

A single server that serves two demos over **one** shared NLLB model instance:

- **Suggest** (``/suggest``): inline autocomplete of the target translation.
- **Evaluate** (``/evaluate``): highlight low-probability spans of a translation
  with click-to-apply alternatives.

The 600M model is loaded exactly once (see :class:`~silnlp.nmt.nllb_demo.model.NllbModel`)
and shared by both :class:`~silnlp.nmt.nllb_demo.suggestion_service.SuggestionService`
and :class:`~silnlp.nmt.nllb_demo.scoring_service.ScoringService`.

The public names below are exposed lazily (PEP 562): importing this package does **not**
pull in torch/transformers. This keeps ``python -m silnlp.nmt.nllb_demo.server`` cheap to
*resolve* — the heavy imports run only when the server actually starts — so launching under
a debugger does not trip debugpy's process-spawn timeout while the parent package imports.
"""

from typing import TYPE_CHECKING, Any

__all__ = ["NllbModel", "ScoringService", "SuggestionService"]

if TYPE_CHECKING:
    from .model import NllbModel
    from .scoring_service import ScoringService
    from .suggestion_service import SuggestionService


def __getattr__(name: str) -> Any:
    if name == "NllbModel":
        from .model import NllbModel

        return NllbModel
    if name == "ScoringService":
        from .scoring_service import ScoringService

        return ScoringService
    if name == "SuggestionService":
        from .suggestion_service import SuggestionService

        return SuggestionService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
