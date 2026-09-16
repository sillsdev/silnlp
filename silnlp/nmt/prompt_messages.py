import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from jinja2.exceptions import UndefinedError
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .model_name import ModelName

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Language:
    iso: str
    name: str


@dataclass
class PromptMessages:
    """The chat messages for a translation prompt: an optional system message, a plain-text
    user instruction, and, for training examples, the target translation as the assistant turn."""

    system_message: str
    instruction: str
    target: Optional[str] = None

    def to_chat_messages(self) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        if self.system_message:
            messages.append({"role": "system", "content": self.system_message})
        messages.append({"role": "user", "content": self.instruction})
        if self.target is not None:
            messages.append({"role": "assistant", "content": self.target})
        return messages

    def to_folded_chat_messages(self) -> List[Dict[str, str]]:
        """Chat messages with the system message folded into the user turn, for chat
        templates that reject a separate system role (e.g. Gemma)."""
        instruction = f"{self.system_message}\n\n{self.instruction}" if self.system_message else self.instruction
        messages: List[Dict[str, str]] = [{"role": "user", "content": instruction}]
        if self.target is not None:
            messages.append({"role": "assistant", "content": self.target})
        return messages

    def to_plain_text(self) -> str:
        return "".join(f"{m['content']}\n" for m in self.to_chat_messages())

    def apply_prompt_template(
        self, tokenizer: PreTrainedTokenizerBase, add_generation_prompt: bool, tokenize: bool
    ) -> Union[str, List[int]]:
        """Apply the model's chat template, with fallbacks for templates that lack a
        system role and for base checkpoints with no chat template at all."""
        if tokenizer.chat_template is not None:
            try:
                return tokenizer.apply_chat_template(
                    self.to_chat_messages(),
                    add_generation_prompt=add_generation_prompt,
                    tokenize=tokenize,
                    return_dict=False,
                )
            except Exception:
                # Some chat templates (e.g. Gemma) reject a separate system role; fold the
                # system message into the first user turn and retry.
                if self.system_message:
                    return tokenizer.apply_chat_template(
                        self.to_folded_chat_messages(),
                        add_generation_prompt=add_generation_prompt,
                        tokenize=tokenize,
                        return_dict=False,
                    )
                raise

        LOGGER.warning(
            "Tokenizer for %s has no chat template; falling back to a plain text prompt.", tokenizer.name_or_path
        )
        text = self.to_plain_text()
        if tokenize:
            return tokenizer(text, add_special_tokens=True)["input_ids"]
        return text


@dataclass(init=False)
class TranslateGemmaPromptMessages(PromptMessages):
    """TranslateGemma's chat template rejects a plain-text user turn: it requires ``content``
    to be a single-item list of {type, source_lang_code, target_lang_code, text}, so this
    subclass carries that structured content instead of a plain-text instruction. It has no
    system message, and -- since TranslateGemma always ships a chat template -- never needs
    the system-message-folding or no-chat-template fallbacks of the base class."""

    source_language: Language
    target_language: Language
    text: str

    def __init__(
        self, source_language: Language, target_language: Language, text: str, target: Optional[str] = None
    ) -> None:
        super().__init__(system_message="", instruction="", target=target)
        self.source_language = source_language
        self.target_language = target_language
        self.text = text

    def to_chat_messages(self) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": self.source_language.iso,
                        "target_lang_code": self.target_language.iso,
                        "text": self.text,
                    }
                ],
            }
        ]
        if self.target is not None:
            messages.append({"role": "assistant", "content": self.target})
        return messages

    def to_folded_chat_messages(self) -> List[Dict[str, str]]:
        raise NotImplementedError("TranslateGemma's structured content is never folded.")

    def to_plain_text(self) -> str:
        raise NotImplementedError("TranslateGemma always has a chat template; there is no plain-text fallback.")

    def apply_prompt_template(
        self, tokenizer: PreTrainedTokenizerBase, add_generation_prompt: bool, tokenize: bool
    ) -> Union[str, List[int]]:
        if tokenizer.chat_template is not None:
            try:
                return tokenizer.apply_chat_template(
                    self.to_chat_messages(),
                    add_generation_prompt=add_generation_prompt,
                    tokenize=tokenize,
                    return_dict=False,
                )
            except UndefinedError:
                # TranslateGemma's template only recognizes its fixed ~55-language lookup table
                # and raises UndefinedError for any other code -- which is the common case when
                # fine-tuning to extend coverage to a new language. Render the same instruction
                # ourselves, using our own configured language name instead of the template's.
                text = self._render_fallback_prompt(tokenizer, add_generation_prompt)
                if tokenize:
                    return tokenizer(text, add_special_tokens=False)["input_ids"]
                return text

        LOGGER.warning(
            "Tokenizer for %s has no chat template; falling back to a plain text prompt.", tokenizer.name_or_path
        )
        text = self._render_fallback_prompt(tokenizer, add_generation_prompt)
        if tokenize:
            return tokenizer(text, add_special_tokens=False)["input_ids"]
        return text

    def _render_fallback_prompt(self, tokenizer: PreTrainedTokenizerBase, add_generation_prompt: bool) -> str:
        """Reimplementation of TranslateGemma's chat template, minus its language-code lookup
        table, for language codes that table doesn't recognize (see apply_prompt_template)."""
        src, trg = self.source_language, self.target_language
        instruction = (
            f"You are a professional {src.name} ({src.iso}) to {trg.name} ({trg.iso}) translator. Your goal is "
            f"to accurately convey the meaning and nuances of the original {src.name} text while adhering to "
            f"{trg.name} grammar, vocabulary, and cultural sensitivities.\n"
            f"Produce only the {trg.name} translation, without any additional explanations or commentary. "
            f"Please translate the following {src.name} text into {trg.name}:\n\n\n{self.text.strip()}"
        )
        text = (tokenizer.bos_token or "") + f"<start_of_turn>user\n{instruction}<end_of_turn>\n"
        if self.target is not None:
            text += f"<start_of_turn>model\n{self.target.strip()}<end_of_turn>\n"
        if add_generation_prompt:
            text += "<start_of_turn>model\n"
        return text


class PromptBuilder:
    """Builds the prompt for one translation, in whichever form the model's chat template expects."""

    def __init__(self, model_name: ModelName, prompt: dict) -> None:
        self._model_name = model_name
        self._prompt = prompt

    def build(
        self, source: str, src_lang: Language, trg_lang: Language, target: Optional[str] = None
    ) -> PromptMessages:
        if self._model_name.uses_translate_gemma_template():
            return TranslateGemmaPromptMessages(
                source_language=src_lang, target_language=trg_lang, text=source, target=target
            )
        instruction = self._prompt["instruction_template"].format(
            src_lang=src_lang.name, trg_lang=trg_lang.name, source=source
        )
        return PromptMessages(self._prompt.get("system_message", ""), instruction, target)
