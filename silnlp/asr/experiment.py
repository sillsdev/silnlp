import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set, Union
import json
import re

import numpy as np
import yaml  # TODO cleanup imports

from silnlp.nmt.clearml_connection import LOGGER, TAGS_LIST, SILClearML  # TODO move clearml to separate module?

from ..common.environment import SIL_NLP_ENV
from ..common.postprocesser import PostprocessConfig, PostprocessHandler
from ..common.utils import get_git_revision_hash, show_attrs
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Union

import evaluate
import pandas as pd
import torch
from datasets import Audio, Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Wav2Vec2BertProcessor,
    Wav2Vec2Processor,
)
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers import (
    AutoProcessor,
    AutoModel,
    AutoFeatureExtractor,
    AutoTokenizer,
    Wav2Vec2CTCTokenizer,
    TrainingArguments,
    Trainer,
    Wav2Vec2FeatureExtractor,
    AutoModelForCTC,
)


@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2BertProcessor | Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"][0]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_attention_mask=True, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def get_vocab(dataset: Dataset, target_language: str):
    def extract_all_chars(batch):
        all_text = " ".join(batch["text"])
        vocab = list(set(all_text))
        return {"vocab": [vocab], "all_text": [all_text]}

    vocab = dataset.map(
        extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=dataset.column_names
    )
    vocab_list = list(set(vocab["vocab"][0]))
    vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    new_vocab_dict = {target_language: vocab_dict}
    with open("vocab.json", "w") as vocab_file:
        json.dump(new_vocab_dict, vocab_file)


def run(experiment_name: str, clearml_queue: str, clearml_tag: str, commit: Optional[str]) -> None:
    clearml = SILClearML(
        experiment_name,
        clearml_queue,
        project_suffix="",
        experiment_suffix="",
        commit=commit,
        tag=clearml_tag,
    )

    # TODO should be in config
    max_input_length = 30.0
    test_size = 0.1
    target_language = "swahili"
    characters_to_remove = ',?.!-;"“%‘”�'

    df1 = pd.read_parquet(
        "/root/M/MT/experiments/demo_ben/misc/w_test/data/senga_ground_truth_transcriptions/train-00000-of-00002.parquet"
    )
    df2 = pd.read_parquet(
        "/root/M/MT/experiments/demo_ben/misc/w_test/data/senga_ground_truth_transcriptions/train-00001-of-00002.parquet"
    )
    df = pd.concat([df1, df2], ignore_index=True)

    dataset = Dataset.from_dict(
        {"text": [row["text"] for _, row in df.iterrows()], "audio": [row["audio"] for _, row in df.iterrows()]}
    )

    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    if len(characters_to_remove) > 0 and not clearml.config.model.startswith("openai/whisper"):
        chars_to_remove_regex = f"[{re.escape(characters_to_remove)}]"

        def remove_characters(batch):
            batch["text"] = re.sub(chars_to_remove_regex, "", batch["text"]).lower()
            return batch

        dataset = dataset.map(remove_characters)

    LOGGER.info(f"Using model {clearml.config.model}")

    if clearml.config.model.startswith("openai/whisper"):
        processor = WhisperProcessor.from_pretrained(clearml.config.model, language=target_language, task="transcribe")
    else:
        get_vocab(dataset, target_language)

        feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True
        )
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
            "./", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|", target_lang=target_language
        )
        processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    if clearml.config.model.startswith("openai/whisper"):
        def prepare_dataset(example):
            audio = example["audio"]

            example = processor(
                audio=audio["array"],
                sampling_rate=audio["sampling_rate"],
                text=example["text"],
            )

            # compute input length of audio sample in seconds
            example["input_length"] = len(audio["array"]) / audio["sampling_rate"]

            return example
    else:
        def prepare_dataset(batch):
            audio = batch["audio"]

            batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
            # batch["input_features"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
            batch["input_length"] = len(batch["input_values"])

            batch["labels"] = processor(text=batch["text"]).input_ids
            return batch

    dataset = dataset.map(prepare_dataset)

    if max_input_length > 0:

        def is_audio_in_length_range(length):
            return length < max_input_length

        dataset = dataset.filter(
            is_audio_in_length_range,
            input_columns=["input_length"],
        )

    from pprint import pformat

    LOGGER.info(pformat(dataset))

    split_dataset = dataset.train_test_split(test_size=test_size)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    if clearml.config.model.startswith("openai/whisper"):
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    else:
        data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        
        if not clearml.config.model.startswith("openai/whisper"):
            pred_ids = np.argmax(pred_ids, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        cer = 100 * cer_metric.compute(predictions=pred_str, references=label_str)

        return {"cer": cer}

    if clearml.config.model.startswith("openai/whisper"):
        model = WhisperForConditionalGeneration.from_pretrained(clearml.config.model)
    else:
        model = AutoModelForCTC.from_pretrained(
            clearml.config.model,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            feat_proj_dropout=0.0,
            layerdrop=0.0,
            ctc_loss_reduction="mean",
            pad_token_id=processor.tokenizer.pad_token_id,
            vocab_size=len(processor.tokenizer),
            ignore_mismatched_sizes=True,
        )

    if clearml.config.model.startswith("openai/whisper"):
        # disable cache during training since it's incompatible with gradient checkpointing
        model.config.use_cache = False

        # set language and task for generation and re-enable cache
        model.generate = partial(model.generate, language=target_language, task="transcribe", use_cache=True)

    elif clearml.config.model.startswith("facebook/mms"):
        model.init_adapter_layers()
        model.freeze_base_model()

        adapter_weights = model._get_adapters()
        for param in adapter_weights.values():
            param.requires_grad = True
    
    if clearml.config.model.startswith("openai/whisper"):
        training_args = Seq2SeqTrainingArguments(
            output_dir="./",
            per_device_train_batch_size=16,
            gradient_accumulation_steps=2,
            learning_rate=1e-5,
            lr_scheduler_type="constant_with_warmup",
            warmup_steps=50,
            max_steps=500,
            gradient_checkpointing=True,
            fp16=True,
            fp16_full_eval=True,
            eval_strategy="steps",
            per_device_eval_batch_size=16,
            predict_with_generate=True,
            generation_max_length=225,
            save_steps=100,
            eval_steps=100,
            logging_steps=25,
            load_best_model_at_end=True,
            metric_for_best_model="cer",
            greater_is_better=False,
        )

        model.generation_config.suppress_tokens = []

        trainer = Seq2SeqTrainer(
            args=training_args,
            model=model.to("cuda"),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            processing_class=processor,
        )
    else:
        training_args = TrainingArguments(
            group_by_length=True,
            per_device_train_batch_size=16,
            gradient_accumulation_steps=2,
            learning_rate=3e-4,
            lr_scheduler_type="constant_with_warmup",
            warmup_steps=50,
            max_steps=500,
            gradient_checkpointing=True,
            fp16=True,
            fp16_full_eval=True,
            eval_strategy="steps",
            per_device_eval_batch_size=16,
            save_steps=100,
            eval_steps=100,
            logging_steps=25,
            load_best_model_at_end=True,
            metric_for_best_model="cer",
            greater_is_better=False,
            push_to_hub=False,
            output_dir="./",
        )

        trainer = Trainer(
            args=training_args,
            model=model.to("cuda"),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            processing_class=processor,
        )

    trainer.train()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiment - preprocess, train, and test")
    parser.add_argument("experiment", help="Experiment name")
    parser.add_argument(
        "--clearml-queue",
        default=None,
        type=str,
        help="Run remotely on ClearML queue.  Default: None - don't register with ClearML.  The queue 'local' will run "
        + "it locally and register it with ClearML.",
    )
    parser.add_argument(
        "--clearml-tag",
        metavar="tag",
        choices=TAGS_LIST,
        default=None,
        type=str,
        help=f"Tag to add to the ClearML Task - {TAGS_LIST}",
    )
    parser.add_argument(
        "--commit", type=str, default=None, help="The silnlp git commit id with which to run a remote job"
    )

    args = parser.parse_args()

    if args.clearml_queue is not None and args.clearml_tag is None:
        parser.error("Missing ClearML tag. Add a tag using --clearml-tag. Possible tags: " + f"{TAGS_LIST}")

    run(args.experiment, args.clearml_queue, args.clearml_tag, args.commit)


if __name__ == "__main__":
    main()
