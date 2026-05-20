import argparse
import logging
from typing import Optional

from .config_utils import load_config
from .translation_scorer import DEFAULT_LOW_PROB_THRESHOLD, DEFAULT_TOP_K_SUGGESTIONS, ScoredTranslation

LOGGER = logging.getLogger(__name__)


def format_scored_translation(scored: ScoredTranslation) -> str:
    """Format a ScoredTranslation as a human-readable string."""
    lines = []
    lines.append(f"Source:      {scored.source}")
    lines.append(f"Translation: {scored.translation}")
    lines.append(f"Overall log-probability: {scored.sequence_log_prob:.4f}")
    lines.append("")

    # Per-word table
    col_word = max(len(w.word) for w in scored.word_scores) if scored.word_scores else 10
    col_word = max(col_word, 4)
    header = f"  {'Word':<{col_word}}  {'Log Prob':>10}  {'Prob':>10}  Suggestions"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for ws in scored.word_scores:
        flag = "* " if ws.is_low_probability else "  "
        suggestions_str = ", ".join(ws.suggestions) if ws.suggestions else ""
        lines.append(
            f"{flag}{ws.word:<{col_word}}  {ws.log_prob:>10.4f}  {ws.prob:>10.6f}  {suggestions_str}"
        )

    lines.append("")
    low_prob = scored.low_probability_words
    if low_prob:
        lines.append("Low-probability words and suggested alternatives:")
        for ws in low_prob:
            if ws.suggestions:
                suggestions_str = ", ".join(f"'{s}'" for s in ws.suggestions)
                lines.append(f"  '{ws.word}'  (log prob {ws.log_prob:.4f})  →  {suggestions_str}")
            else:
                lines.append(f"  '{ws.word}'  (log prob {ws.log_prob:.4f})  →  (no suggestions available)")
    else:
        lines.append("No low-probability words found.")

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
    """Score a translation against a source sentence using a trained NMT model.

    Loads the experiment's model, runs forced decoding on the translation, and returns
    a ScoredTranslation with per-word probabilities and suggestions for flagged words.

    Args:
        experiment: Name of the experiment (relative to the MT experiments directory).
        source: The source sentence to score against.
        translation: The translation to evaluate.
        src_iso: Source language ISO code. Defaults to the experiment's test source.
        trg_iso: Target language ISO code. Defaults to the experiment's test target.
        checkpoint: Checkpoint to load ("last", "best", "avg", or a step number).
        low_prob_threshold: Log-probability threshold for flagging low-probability words.
        top_k_suggestions: Number of alternative suggestions per flagged word.

    Returns:
        A ScoredTranslation with per-word scores and suggestions.
    """
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
            "Score a translation by computing the model's token-level conditional probabilities. "
            "Low-probability words are flagged and paired with suggested alternatives from the model."
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
            f"Log-probability threshold below which a word is considered low-probability "
            f"(default: {DEFAULT_LOW_PROB_THRESHOLD})"
        ),
    )
    parser.add_argument(
        "--top-k-suggestions",
        type=int,
        default=DEFAULT_TOP_K_SUGGESTIONS,
        help=f"Number of alternative suggestions per low-probability word (default: {DEFAULT_TOP_K_SUGGESTIONS})",
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
