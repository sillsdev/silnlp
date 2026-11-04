from pathlib import Path
from typing import Iterable, List, Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ..common.translation_data_structures import SentenceTranslation, SentenceTranslationGroup
from .causal_lm_provider import CausalLMProvider
from .causal_lm_tokenizer import CausalLMTokenizer
from .example_retrieval import ExamplePools
from .experiment_languages import Language
from .generation_settings import GenerationSettings
from .local_llm_prompts import LocalLLMPrompts
from .seq2seq_config import batch_sentences


class CausalLMGenerator:
    """Translates with a decoder-only model: one prompt per sentence, and one completion per draft."""

    def __init__(
        self,
        provider: CausalLMProvider,
        tokenizer: CausalLMTokenizer,
        prompts: LocalLLMPrompts,
        settings: GenerationSettings,
        example_pools: ExamplePools,
    ) -> None:
        self._provider = provider
        self._tokenizer = tokenizer
        self._prompts = prompts
        self._settings = settings
        self._example_pools = example_pools

    def ensure_examples_available(self) -> None:
        self._example_pools.ensure_available()

    def load(self, checkpoint_path: Optional[Path]) -> PreTrainedModel:
        model = self._provider.create_model_for_inference(checkpoint_path)
        if torch.cuda.is_available():
            model = model.to("cuda")
        model.eval()
        return model

    def loaded_tokenizer(self) -> PreTrainedTokenizerBase:
        return self._tokenizer.load()

    def generate(
        self,
        model: PreTrainedModel,
        sentences: Iterable[str],
        src_lang: Language,
        trg_lang: Language,
        num_return_sequences: int,
        save_confidences: bool,
    ) -> Iterable[SentenceTranslationGroup]:
        tokenizer = self._tokenizer.load()
        tokenizer.padding_side = "left"
        gen_kwargs = self._settings.as_keyword_arguments(num_return_sequences, tokenizer.pad_token_id)

        device = model.device
        for batch in batch_sentences(sentences, self._settings.batch_size()):
            prompts = [
                self._prompts.build(sentence, src_lang, trg_lang).apply_prompt_template(
                    tokenizer, add_generation_prompt=True, tokenize=False
                )
                for sentence in batch
            ]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    return_dict_in_generate=True,
                    output_scores=save_confidences,
                    **gen_kwargs,
                )
            prompt_length = inputs["input_ids"].shape[1]
            generated = output.sequences[:, prompt_length:]

            transition_scores = None
            beam_indices = None
            if save_confidences and getattr(output, "scores", None) is not None:
                beam_indices = getattr(output, "beam_indices", None)
                transition_scores = model.compute_transition_scores(
                    output.sequences, output.scores, beam_indices, normalize_logits=True
                )

            for i in range(len(batch)):
                translations: List[SentenceTranslation] = []
                for j in range(num_return_sequences):
                    seq_index = i * num_return_sequences + j
                    token_ids = generated[seq_index]
                    text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
                    token_scores: List[float] = []
                    sequence_score: Optional[float] = None
                    if transition_scores is not None:
                        scores_row = transition_scores[seq_index].tolist()
                        if beam_indices is not None:
                            # With beam search, compute_transition_scores() marks padded positions
                            # with a beam index of -1 and a transition score of 0.
                            valid = [s for s, b in zip(scores_row, beam_indices[seq_index].tolist()) if b >= 0]
                        else:
                            valid = [s for s in scores_row if s != float("-inf")]
                        token_scores = valid
                        if len(valid) > 0:
                            sequence_score = sum(valid) / len(valid)
                    translations.append(
                        SentenceTranslation(text, [text], token_scores, sequence_score, starts_with_special_token=False)
                    )
                yield SentenceTranslationGroup(translations)
