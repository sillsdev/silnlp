"""Ensemble decoding inside HF beam search: several (model, source text) members vote on every next token.

`generate` drives one member as usual.  The others sit in a LogitsProcessor, follow the same hypotheses, and their
next-token distributions are combined with the driver's.  Members may be different models with one target vocabulary,
or one model reading different source texts.
"""

import math

import torch
from transformers import LogitsProcessor
from transformers.modeling_outputs import BaseModelOutput


class Member:
    """One model reading one batch of source texts, scoring the next token for the hypotheses beam search holds."""

    def __init__(self, model, input_ids: torch.Tensor, attention_mask: torch.Tensor, num_beams: int) -> None:
        self._model = model
        self._num_beams = num_beams
        with torch.no_grad():
            encoded = model.get_encoder()(input_ids=input_ids, attention_mask=attention_mask)
        hidden = encoded.last_hidden_state.repeat_interleave(num_beams, dim=0)
        self._encoder_outputs = BaseModelOutput(last_hidden_state=hidden)
        self._attention_mask = attention_mask.repeat_interleave(num_beams, dim=0)
        self._past = None
        self._prefixes = None

    def next_token_log_probs(self, prefixes: torch.Tensor) -> torch.Tensor:
        decoder_input_ids = prefixes
        if self._past is not None:
            self._past.reorder_cache(self._parents(prefixes))
            decoder_input_ids = prefixes[:, -1:]
        with torch.no_grad():
            output = self._model(
                encoder_outputs=self._encoder_outputs,
                attention_mask=self._attention_mask,
                decoder_input_ids=decoder_input_ids,
                past_key_values=self._past,
                use_cache=True,
            )
        self._past = output.past_key_values
        self._prefixes = prefixes.clone()
        return torch.log_softmax(output.logits[:, -1].float(), dim=-1)

    def _parents(self, prefixes: torch.Tensor) -> torch.Tensor:
        # Beam search reorders its rows between steps without saying how, so each row's cached decoder state is
        # found by matching its prefix against the rows of the previous step, within the same sentence.
        rows = torch.arange(prefixes.shape[0], device=prefixes.device)
        same_sentence = (rows[:, None] // self._num_beams) == (rows[None, :] // self._num_beams)
        extends = (prefixes[:, None, :-1] == self._prefixes[None, :, :]).all(-1) & same_sentence
        if not bool(extends.any(-1).all()):
            raise RuntimeError("a hypothesis does not extend any hypothesis of the previous step")
        return extends.int().argmax(-1)


class Ensemble(LogitsProcessor):
    """Replaces the driving model's next-token log-probabilities with a weighted combination over all members.

    rule "linear" averages probabilities (a token needs support from any member); "loglinear" averages
    log-probabilities and renormalises (a token needs support from every member).
    """

    def __init__(self, others: list[Member], driver_weight: float, other_weights: list[float], rule: str) -> None:
        self._others = others
        self._weights = [driver_weight] + other_weights
        self._rule = rule

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        log_probs = [scores] + [member.next_token_log_probs(input_ids) for member in self._others]
        voting = [(w, lp) for w, lp in zip(self._weights, log_probs) if w > 0]
        if self._rule == "linear":
            combined = torch.logsumexp(torch.stack([lp + math.log(w) for w, lp in voting]), dim=0)
        else:
            combined = torch.log_softmax(sum(w * lp for w, lp in voting), dim=-1)
        ruled_out = torch.isneginf(scores)
        combined = combined.masked_fill(ruled_out, float("-inf"))
        # Where the driver's processors force one token (the language code, the end at the length limit) the step
        # costs what they say: a member's own probability of that token would shift every hypothesis by a constant.
        forced = (~ruled_out).sum(-1) == 1
        return torch.where(forced[:, None], scores, combined)
