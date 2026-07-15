import argparse
import logging
from typing import Optional

from .config_utils import load_config
from .translation_scorer import DEFAULT_LOW_PROB_THRESHOLD, DEFAULT_TOP_K_SUGGESTIONS, ScoredTranslation, SpanScore

LOGGER = logging.getLogger(__name__)


def _format_suggestions(span: SpanScore) -> str:
    if not span.suggestions:
        return ""
    return "; ".join(
        f"{suggestion.phrase} (logp/tok={suggestion.mean_token_log_prob:.4f}, Δ={suggestion.improvement:.4f})"
        for suggestion in span.suggestions
    )


def format_scored_translation(scored: ScoredTranslation) -> str:
    """Format a ScoredTranslation as a human-readable string."""
    lines = [
        f"Source:      {scored.source}",
        f"Translation: {scored.translation}",
        f"Total log-probability:   {scored.total_log_prob:.4f}",
        f"Mean log-prob per token: {scored.mean_token_log_prob:.4f}",
        "",
        "Word-level scores:",
    ]

    col_word = max((len(score.text) for score in scored.word_scores), default=4)
    header = f"  {'Word':<{col_word}}  {'Forward':>10}  {'Right Ctx':>10}  {'LogP/Tok':>10}  Suggestions"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for score in scored.word_scores:
        flag = "* " if score.is_flagged else "  "
        suggestion_text = ", ".join(suggestion.phrase for suggestion in score.suggestions)
        lines.append(
            f"{flag}{score.text:<{col_word}}  {score.forward_log_prob:>10.4f}  "
            f"{score.right_context_log_prob:>10.4f}  {score.mean_token_log_prob:>10.4f}  {suggestion_text}"
        )

    lines.append("")
    lines.append("Flagged phrases:")
    flagged_phrases = scored.flagged_phrases
    if not flagged_phrases:
        lines.append("  None")
    else:
        for score in flagged_phrases:
            lines.append(
                f"  '{score.text}' [{score.word_start}:{score.word_end}] "
                f"forward={score.forward_log_prob:.4f}, right_ctx={score.right_context_log_prob:.4f}, "
                f"logp/tok={score.mean_token_log_prob:.4f}"
            )
            suggestions = _format_suggestions(score)
            if suggestions:
                lines.append(f"    Suggestions: {suggestions}")

    return "\n".join(lines)


def score_translation(
    experiment: str,
    source: str,
    translation: str,
    src_iso: Optional[str],
    trg_iso: Optional[str],
    checkpoint: str = "last",
    low_prob_threshold: float = DEFAULT_LOW_PROB_THRESHOLD,
    top_k_suggestions: int = DEFAULT_TOP_K_SUGGESTIONS,
) -> ScoredTranslation:
    config = load_config(experiment)
    model = config.create_model()

    effective_src_iso = src_iso or config.default_test_src_iso
    effective_trg_iso = trg_iso or config.default_test_trg_iso

    return model.score_translation(
        source=source,
        translation=translation,
        src_iso=effective_src_iso,
        trg_iso=effective_trg_iso,
        ckpt=checkpoint,
        low_prob_threshold=low_prob_threshold,
        top_k_suggestions=top_k_suggestions,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Score a translation by computing contextual phrase probabilities. "
            "Low-probability words and phrases are flagged and paired with rescored replacement suggestions."
        )
    )
    parser.add_argument("experiment", help="Experiment name")
    parser.add_argument("--source", type=str, required=True, help="Source sentence to score against")
    parser.add_argument("--translation", type=str, required=True, help="Translation to evaluate")
    parser.add_argument("--src-iso", type=str, default=None, help="Source language ISO code")
    parser.add_argument("--trg-iso", type=str, default=None, help="Target language ISO code")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="last",
        help="Checkpoint to use: 'last', 'best', 'avg', or a checkpoint step number",
    )
    parser.add_argument(
        "--low-prob-threshold",
        type=float,
        default=DEFAULT_LOW_PROB_THRESHOLD,
        help=(
            f"Mean per-token log-probability below which a span is flagged " f"(default: {DEFAULT_LOW_PROB_THRESHOLD})"
        ),
    )
    parser.add_argument(
        "--top-k-suggestions",
        type=int,
        default=DEFAULT_TOP_K_SUGGESTIONS,
        help=f"Number of replacement suggestions per flagged span (default: {DEFAULT_TOP_K_SUGGESTIONS})",
    )

    args = parser.parse_args()

    scored = score_translation(
        experiment=args.experiment,
        source=args.source,
        translation=args.translation,
        src_iso=args.src_iso,
        trg_iso=args.trg_iso,
        checkpoint=args.checkpoint,
        low_prob_threshold=args.low_prob_threshold,
        top_k_suggestions=args.top_k_suggestions,
    )
    print(format_scored_translation(scored))


if __name__ == "__main__":
    main()
