import logging
from typing import Any, Optional, Tuple, Union

from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.m2m_100.tokenization_m2m_100 import M2M100Tokenizer
from transformers.models.mbart.tokenization_mbart import MBartTokenizer
from transformers.models.mbart50.tokenization_mbart50 import MBart50Tokenizer
from transformers.models.nllb.tokenization_nllb import NllbTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .checkpoints import CheckpointDirectory, CheckpointType
from .model_name import ModelName
from .pretrained_tokenizer import PretrainedTokenizer
from .tokenizer_settings import TokenizerSettings
from .translation_settings import ModelSettings

LOGGER = logging.getLogger(__name__)


class PretrainedModelLoader:
    """The model an experiment runs on, configured for the languages it translates between: either a
    fresh one to train, or the one a checkpoint holds."""

    # Splitting NLLB across two devices puts the encoder on the first and most of the decoder on the second.
    _NLLB_TWO_DEVICE_MAP = {
        "lm_head": 0,
        "model.shared": 0,
        "model.encoder": 0,
        "model.decoder.embed_tokens": 0,
        "model.decoder.embed_positions": 1,
        "model.decoder.layers": 1,
        "model.decoder.layer_norm": 1,
    }

    def __init__(
        self,
        provider: Any,
        model: str,
        model_name: ModelName,
        pretrained_tokenizer: PretrainedTokenizer,
        tokenizer_settings: TokenizerSettings,
        model_settings: ModelSettings,
        checkpoints: CheckpointDirectory,
        num_devices: int,
    ) -> None:
        self._provider = provider
        self._model = model
        self._model_name = model_name
        self._pretrained_tokenizer = pretrained_tokenizer
        self._tokenizer_settings = tokenizer_settings
        self._model_settings = model_settings
        self._checkpoints = checkpoints
        self._num_devices = num_devices

    def for_training(
        self, training_args: Any, src_lang: str, trg_lang: str
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        model_config = AutoConfig.from_pretrained(
            self._model,
            use_cache=not training_args.gradient_checkpointing,
            dropout=self._model_settings.dropout(),
            attention_dropout=self._model_settings.attention_dropout(),
            activation_dropout=self._model_settings.activation_dropout(),
            label2id={},
            id2label={},
            num_labels=0,
            attn_implementation=self._model_settings.attention_implementation(),
            token=False,
        )
        model = self._provider.create_model_for_training(
            self._model, model_config, device_map=self._device_map()
        )
        tokenizer = self._pretrained_tokenizer.load()
        self._fit_embeddings_to(model, tokenizer, training_args)
        return self.configured(model, tokenizer, src_lang, trg_lang)

    def for_inference(
        self, ckpt: Union[CheckpointType, str, int], src_lang: str, trg_lang: str
    ) -> PreTrainedModel:
        if self._checkpoints.exists():
            model_name = str(self._checkpoints.resolve(ckpt).path)
        else:
            LOGGER.warning("Model has no checkpoints. Using base model.")
            model_name = self._model

        model: PreTrainedModel = self._provider.create_model_for_inference(model_name)
        model, _ = self.configured(model, self._pretrained_tokenizer.load(), src_lang, trg_lang)

        if model.generation_config is not None and (
            model.generation_config.max_length is None or model.generation_config.max_length < 512
        ):
            model.generation_config.max_length = 512

        return model

    def configured(
        self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, src_lang: str, trg_lang: str
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        if trg_lang != "" and model.config.decoder_start_token_id is None and isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(trg_lang)

        if self._model_name.is_madlad():
            model.config.decoder_start_token_id = tokenizer.pad_token_id
            model.generation_config.decoder_start_token_id = tokenizer.pad_token_id
            model.generation_config.max_length = 256
            model.generation_config.max_new_tokens = 256
            tokenizer.tgt_lang = trg_lang

        if model.config.decoder_start_token_id is None:
            raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

        multilingual = isinstance(tokenizer, (MBartTokenizer, MBart50Tokenizer, M2M100Tokenizer, NllbTokenizer))
        if src_lang != "" and trg_lang != "" and multilingual:
            tokenizer.src_lang = src_lang
            tokenizer.tgt_lang = trg_lang

            # For multilingual translation models like mBART-50 and M2M100 we need to force the target language token
            # as the first generated token.
            forced_bos_token_id = tokenizer.convert_tokens_to_ids(trg_lang)
            if model.generation_config is not None:
                model.generation_config.forced_bos_token_id = forced_bos_token_id

        if len(tokenizer) > model.get_input_embeddings().weight.size(dim=0):
            # NOTE: This is only a warning because the smoke tests use a mismatched tokenizer and model (intentionally).
            # The long-term fix for this is to use dependency injection for the tokenizer
            LOGGER.warning(
                f"Tokenizer vocab size ({len(tokenizer)}) does not match the model's embedding vocab size "
                f"({model.get_input_embeddings().weight.size(dim=0)}). Ensure you are using the correct "
                f"tokenizer for this checkpoint."
            )

        return model, tokenizer

    def _device_map(self) -> Optional[dict]:
        if self._num_devices == 2 and self._model_name.is_nllb():
            return dict(self._NLLB_TWO_DEVICE_MAP)
        return None

    def _fit_embeddings_to(
        self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, training_args: Any
    ) -> None:
        old_embeddings = model.get_input_embeddings()
        old_num_tokens = old_embeddings.weight.size(dim=0)
        if len(tokenizer) <= old_num_tokens:
            return
        pad_to_multiple_of = 8 if training_args.fp16 or training_args.bf16 else None
        if self._tokenizer_settings.initializes_unknown():
            vocab = tokenizer.get_vocab()
            unk_embedding = old_embeddings.weight.data[vocab["<unk>"]]
            model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=pad_to_multiple_of)
            embeddings = model.get_input_embeddings()
            embeddings.weight.data[old_num_tokens:, :] = unk_embedding
            model.tie_weights()
        else:
            model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=pad_to_multiple_of)
