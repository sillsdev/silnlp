"""The single shared NLLB model used by both demos.

``NllbModel`` is the sole owner of the loaded weights: it performs exactly one
``from_pretrained`` pair (tokenizer + model) and both services hold a reference to
this object. Nothing else in the package loads a model, so the 600M weights live in
memory only once regardless of how many demos are served.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from silnlp.common.iso_info import NLLB_TAGS

from ..hugging_face_config import ConstraintIndexes, load_partial_word_constraint_indexes

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "facebook/nllb-200-distilled-600M"
DEFAULT_SOURCE_LANG = "eng_Latn"
DEFAULT_TARGET_LANG = "fra_Latn"


@dataclass
class NllbModel:
    """A single, shared NLLB model + tokenizer.

    All access to :attr:`model` / :attr:`tokenizer` must happen while holding
    :attr:`lock`, because both demos mutate shared state (``tokenizer.src_lang`` /
    ``tokenizer.tgt_lang`` and ``model.config.forced_bos_token_id``) at the start of
    every request.
    """

    model: Any
    tokenizer: Any
    device: torch.device
    language_codes: List[str]
    lock: threading.Lock = field(default_factory=threading.Lock)
    _constraint_indexes: Optional[ConstraintIndexes] = None

    @classmethod
    def load(cls, model_name: str = DEFAULT_MODEL_NAME) -> "NllbModel":
        """Load the tokenizer and model exactly once.

        Uses fp16 on CUDA (to fit the demo comfortably in VRAM) and fp32 on CPU
        (fp16 kernels are poorly supported there).
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
        LOGGER.info("Loading %s (dtype=%s)", model_name, torch_dtype)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch_dtype)
        model.to(device)
        model.eval()
        LOGGER.info("Loaded %s on %s", model_name, device)
        return cls(
            model=model,
            tokenizer=tokenizer,
            device=device,
            language_codes=sorted(set(NLLB_TAGS)),
        )

    @property
    def constraint_indexes(self) -> ConstraintIndexes:
        """Partial-word constraint indexes, built lazily on first use."""
        if self._constraint_indexes is None:
            self._constraint_indexes = load_partial_word_constraint_indexes(self.tokenizer)
        return self._constraint_indexes

    def supports_language(self, code: str) -> bool:
        return code in self.language_codes

    def warmup(self, source_lang: str = DEFAULT_SOURCE_LANG, target_lang: str = DEFAULT_TARGET_LANG) -> None:
        """Run a dummy forward + generate pass so CUDA kernels compile at startup.

        The first CUDA forward pass triggers JIT kernel compilation and cuDNN
        auto-tuning (15-20 s on some hardware). Doing it here moves that cost off the
        first real request. No-op on CPU.
        """
        if self.device.type != "cuda":
            return
        from transformers.modeling_outputs import BaseModelOutput

        with self.lock:
            LOGGER.info("Warming up CUDA kernels...")
            t0 = time.perf_counter()
            self.tokenizer.src_lang = source_lang
            self.tokenizer.tgt_lang = target_lang
            src = self.tokenizer("warmup", return_tensors="pt")
            tgt = self.tokenizer(text_target="warmup", return_tensors="pt")
            input_ids = src["input_ids"].to(self.device)
            attention_mask = src["attention_mask"].to(self.device)
            labels = tgt["input_ids"].to(self.device)
            with torch.no_grad():
                encoder_out = self.model.get_encoder()(input_ids=input_ids, attention_mask=attention_mask)
                self.model(encoder_outputs=encoder_out, attention_mask=attention_mask, labels=labels)
                self.model.generate(
                    encoder_outputs=BaseModelOutput(last_hidden_state=encoder_out.last_hidden_state),
                    attention_mask=attention_mask,
                    num_beams=4,
                    max_new_tokens=4,
                )
            torch.cuda.synchronize(self.device)
            LOGGER.info("Warmup complete in %.2f s", time.perf_counter() - t0)
