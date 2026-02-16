from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Union

import evaluate
import pandas as pd
import torch
from datasets import Audio, Dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

from silnlp.common.clearml_connection import SILClearML

clearml = SILClearML(
    "demo_ben/misc/w_test",
    "jobs_backlog",
    project_suffix="",
    experiment_suffix="",
    commit=None,
    tag="research",
)

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

processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="swahili", task="transcribe")


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


dataset = dataset.map(prepare_dataset)  # , num_proc=1

max_input_length = 30.0


def is_audio_in_length_range(length):
    return length < max_input_length


dataset = dataset.filter(
    is_audio_in_length_range,
    input_columns=["input_length"],
)

split_dataset = dataset.train_test_split(test_size=0.1)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

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


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

metric = evaluate.load("cer")

normalizer = BasicTextNormalizer()


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    # compute orthographic wer
    cer_ortho = 100 * metric.compute(predictions=pred_str, references=label_str)

    # compute normalised CER
    pred_str_norm = [normalizer(pred) for pred in pred_str]
    label_str_norm = [normalizer(label) for label in label_str]
    # filtering step to only evaluate the samples that correspond to non-zero references:
    pred_str_norm = [pred_str_norm[i] for i in range(len(pred_str_norm)) if len(label_str_norm[i]) > 0]
    label_str_norm = [label_str_norm[i] for i in range(len(label_str_norm)) if len(label_str_norm[i]) > 0]

    cer = 100 * metric.compute(predictions=pred_str_norm, references=label_str_norm)

    return {"cer_ortho": cer_ortho, "cer": cer}


model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

# disable cache during training since it's incompatible with gradient checkpointing
model.config.use_cache = False

# set language and task for generation and re-enable cache
model.generate = partial(model.generate, language="swahili", task="transcribe", use_cache=True)

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-dv",  # name on the HF Hub
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=50,
    max_steps=500,  # increase to 4000 if you have your own GPU or a Colab paid plan
    gradient_checkpointing=True,
    fp16=True,
    fp16_full_eval=True,
    eval_strategy="steps",
    per_device_eval_batch_size=16,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=500,
    eval_steps=500,
    logging_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="cer",
    greater_is_better=False,
    push_to_hub=False,
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

trainer.train()
