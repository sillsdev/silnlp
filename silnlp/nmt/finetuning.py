from typing import Any

from transformers.modeling_utils import PreTrainedModel

from .experiment_settings import TrainerSettings
from .finetune_method import FinetuneMethod


class Finetuning:
    """How a decoder-only model is prepared for training: in full, or wrapped in a low-rank adapter that
    leaves the pretrained weights alone."""

    def __init__(self, method: FinetuneMethod, adapter: dict, trainer_settings: TrainerSettings) -> None:
        self._method = method
        self._adapter = adapter
        self._trainer_settings = trainer_settings

    def applied_to(self, model: PreTrainedModel) -> PreTrainedModel:
        if self._method.is_full():
            return model

        from peft import get_peft_model, prepare_model_for_kbit_training

        gradient_checkpointing = self._trainer_settings.checkpoints_gradients()
        if self._method.uses_quantization():
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)
        elif gradient_checkpointing:
            model.enable_input_require_grads()

        model = get_peft_model(model, self.adapter_config())
        model.print_trainable_parameters()
        return model

    def adapter_config(self) -> Any:
        from peft import LoraConfig, TaskType

        return LoraConfig(
            r=self._adapter["rank"],
            lora_alpha=self._adapter["alpha"],
            lora_dropout=self._adapter["dropout"],
            target_modules=self._adapter["target_modules"],
            modules_to_save=self._adapter.get("modules_to_save"),
            use_dora=self._method.uses_dora(),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
