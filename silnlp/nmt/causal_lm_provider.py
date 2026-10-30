from pathlib import Path
from typing import Any, Dict, Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, PreTrainedModel

from .experiment_settings import TrainerSettings
from .finetune_method import FinetuneMethod


class CausalLMModelSettings:
    """The params section as it bears on loading a decoder-only model's weights."""

    def __init__(self, params: dict) -> None:
        self._params = params

    def dtype(self, mixed_precision: bool) -> Any:
        if not mixed_precision:
            return "auto"
        return getattr(torch, self._params["torch_dtype"], torch.bfloat16)

    def load_arguments(self, mixed_precision: bool) -> Dict[str, Any]:
        return dict(
            torch_dtype=self.dtype(mixed_precision),
            attn_implementation=self._params["attn_implementation"],
            trust_remote_code=self._params["trust_remote_code"],
        )

    def trusts_remote_code(self) -> bool:
        return self._params["trust_remote_code"]

    def precision_arguments(self, mixed_precision: bool) -> Dict[str, bool]:
        dtype = self._params["torch_dtype"]
        return {
            "bf16": mixed_precision and dtype == "bfloat16",
            "fp16": mixed_precision and dtype == "float16",
        }


class CausalLMProvider:
    """Loads the underlying causal LM for training and inference. Indirected so tests can
    substitute a mock provider (mirrors PreTrainedModelProvider in seq2seq_config.py)."""

    def __init__(
        self,
        model: str,
        settings: CausalLMModelSettings,
        finetuning: FinetuneMethod,
        trainer_settings: TrainerSettings,
        mixed_precision: bool,
    ) -> None:
        self._model = model
        self._settings = settings
        self._finetuning = finetuning
        self._trainer_settings = trainer_settings
        self._mixed_precision = mixed_precision

    def _is_image_text_to_text_model(self, model_name_or_path: str) -> bool:
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=self._settings.trusts_remote_code())
        return type(config) in AutoModelForImageTextToText._model_mapping

    def _determine_auto_model_class(self, model_name_or_path: str) -> type:
        if self._is_image_text_to_text_model(model_name_or_path):
            return AutoModelForImageTextToText
        return AutoModelForCausalLM

    def _set_use_cache(self, model: PreTrainedModel, use_cache: bool) -> None:
        # Composite configs (e.g. Gemma3's image-text-to-text wrapper) only expose use_cache on
        # the nested text_config, not on the top-level config, so passing use_cache directly to
        # from_pretrained() leaves it unconsumed there and it gets forwarded as an invalid
        # constructor kwarg to models whose __init__ takes only `config`.
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = use_cache
        text_config = getattr(model.config, "text_config", None)
        if text_config is not None and hasattr(text_config, "use_cache"):
            text_config.use_cache = use_cache

    def create_model_for_training(self) -> PreTrainedModel:
        quantization_config = None
        device_map = None
        if self._finetuning.uses_quantization():
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self._settings.dtype(self._mixed_precision),
                bnb_4bit_use_double_quant=True,
            )
            device_map = {"": 0}
        model_class = self._determine_auto_model_class(self._model)
        model = model_class.from_pretrained(
            self._model,
            quantization_config=quantization_config,
            device_map=device_map,
            **self._settings.load_arguments(self._mixed_precision),
        )
        self._set_use_cache(model, not self._trainer_settings.checkpoints_gradients())
        return model

    def create_model_for_inference(self, checkpoint_path: Optional[Path]) -> PreTrainedModel:
        load_kwargs = self._settings.load_arguments(self._mixed_precision)
        if checkpoint_path is None:
            model_class = self._determine_auto_model_class(self._model)
            return model_class.from_pretrained(self._model, **load_kwargs)

        if (checkpoint_path / "adapter_config.json").is_file():
            from peft import PeftModel

            model_class = self._determine_auto_model_class(self._model)
            base_model = model_class.from_pretrained(self._model, **load_kwargs)
            base_dtype = next(base_model.parameters()).dtype
            model = PeftModel.from_pretrained(base_model, str(checkpoint_path))
            merged = model.merge_and_unload()
            return merged.to(base_dtype)
        model_class = self._determine_auto_model_class(str(checkpoint_path))
        return model_class.from_pretrained(str(checkpoint_path), **load_kwargs)


class CausalLMProviderFactory:
    def create(
        self,
        model: str,
        settings: CausalLMModelSettings,
        finetuning: FinetuneMethod,
        trainer_settings: TrainerSettings,
        mixed_precision: bool,
    ) -> CausalLMProvider:
        raise NotImplementedError


class FileCausalLMProviderFactory(CausalLMProviderFactory):
    def create(
        self,
        model: str,
        settings: CausalLMModelSettings,
        finetuning: FinetuneMethod,
        trainer_settings: TrainerSettings,
        mixed_precision: bool,
    ) -> CausalLMProvider:
        return CausalLMProvider(model, settings, finetuning, trainer_settings, mixed_precision)
