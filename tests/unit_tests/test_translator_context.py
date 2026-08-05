from pathlib import Path
from typing import Generator, Iterable, List

from silnlp.common.environment import SilNlpEnv
from silnlp.common.sentence_context import CONTEXT_END_TOKEN, CONTEXT_START_TOKEN
from silnlp.common.translation_data_structures import SentenceTranslation, SentenceTranslationGroup
from silnlp.common.translator import Translator

SENTENCES = ["one.", "two.", "three.", "four."]


class EchoTranslator(Translator):
    """Echoes each input sentence back as its translation, recording what it was asked to translate."""

    def __init__(self, context_size: int = 0):
        super().__init__(SilNlpEnv.create_standard_environment(), context_size)
        self.received: List[str] = []

    def translate(
        self,
        sentences: Iterable[str],
        src_iso: str,
        trg_iso: str,
        produce_multiple_translations: bool = False,
    ) -> Generator[SentenceTranslationGroup, None, None]:
        for sentence in sentences:
            self.received.append(sentence)
            tokens = sentence.split()
            yield SentenceTranslationGroup(
                [SentenceTranslation(sentence, tokens, [0.0] * len(tokens), 0.0, starts_with_special_token=False)]
            )


def translate_text_file(tmp_path: Path, context_size: int) -> tuple[EchoTranslator, List[str]]:
    src_path = tmp_path / "src.txt"
    trg_path = tmp_path / "trg.txt"
    src_path.write_text("\n".join(SENTENCES) + "\n", encoding="utf-8")

    translator = EchoTranslator(context_size)
    translator.translate_text(src_path, trg_path, "en", "es")
    return translator, trg_path.read_text(encoding="utf-8").splitlines()


def test_input_sentences_arrive_surrounded_by_their_neighbors(tmp_path: Path):
    translator, _ = translate_text_file(tmp_path, context_size=1)
    assert translator.received == [
        f"{CONTEXT_START_TOKEN} one. {CONTEXT_END_TOKEN} two.",
        f"one. {CONTEXT_START_TOKEN} two. {CONTEXT_END_TOKEN} three.",
        f"two. {CONTEXT_START_TOKEN} three. {CONTEXT_END_TOKEN} four.",
        f"three. {CONTEXT_START_TOKEN} four. {CONTEXT_END_TOKEN}",
    ]


def test_the_draft_holds_only_the_translated_sentences(tmp_path: Path):
    _, output_lines = translate_text_file(tmp_path, context_size=1)
    assert output_lines == SENTENCES


def test_there_is_one_output_line_per_input_line(tmp_path: Path):
    for context_size in (0, 1, 2, 5):
        translator, output_lines = translate_text_file(tmp_path, context_size)
        assert len(output_lines) == len(SENTENCES)
        assert len(translator.received) == len(SENTENCES)


def test_sentences_are_translated_one_at_a_time_when_context_is_off(tmp_path: Path):
    translator, output_lines = translate_text_file(tmp_path, context_size=0)
    assert translator.received == SENTENCES
    assert output_lines == SENTENCES
