import argparse
import copy
import logging
import os
from glob import glob
from typing import IO, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Text, Tuple

logging.basicConfig()

import numpy as np
import sacrebleu
import sentencepiece as sp
import tensorflow as tf
from lit_nlp import dev_server, server_flags
from lit_nlp.api import components as lit_components
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
from lit_nlp.components import index, metrics, pca, projection, similarity_searcher, umap, word_replacer
from lit_nlp.lib import caching
from opennmt.runner import _CONFIG_FALLBACK
from opennmt.utils.checkpoint import Checkpoint
from opennmt.utils.misc import clone_layer, extract_batches, merge_dict
from tensorflow.python.eager.def_function import Function

from ..common.utils import get_git_revision_hash, get_mt_root_dir
from .config import Language, create_model, load_config, parse_langs, set_log_level
from .runner import make_inference_dataset
from .transformer import SILTransformer
from .utils import decode_sp, encode_sp, get_best_model_dir, get_last_checkpoint


def create_test_dataset(
    root_dir: str, src_langs: Dict[str, Language], trg_langs: Dict[str, Language]
) -> lit_dataset.Dataset:
    vref_paths: List[str] = []
    features_paths: List[str] = []
    refs_paths: List[str] = []
    for src_iso in sorted(src_langs.keys()):
        prefix = "test" if len(src_langs) == 1 else f"test.{src_iso}"
        src_features_path = os.path.join(root_dir, f"{prefix}.src.txt")
        if os.path.isfile(src_features_path):
            # all target data is stored in a single file
            vref_paths.append(os.path.join(root_dir, f"{prefix}.vref.txt"))
            features_paths.append(src_features_path)
            refs_paths.append(os.path.join(root_dir, f"{prefix}.trg.detok*.txt"))
        else:
            # target data is split into separate files
            for trg_iso in sorted(trg_langs.keys()):
                prefix = f"test.{src_iso}.{trg_iso}"
                vref_paths.append(os.path.join(root_dir, f"{prefix}.vref.txt"))
                features_paths.append(os.path.join(root_dir, f"{prefix}.src.txt"))
                refs_paths.append(os.path.join(root_dir, f"{prefix}.trg.detok*.txt"))

    default_src_iso = next(iter(src_langs.keys()))
    default_trg_iso = next(iter(trg_langs.keys()))

    spec = lit_types.JsonDict = {
        "vref": lit_types.CategoryLabel(),
        "src_text": lit_types.TextSegment(),
        "ref_text": lit_types.TextSegment(),
        "src_iso": lit_types.CategoryLabel(),
        "trg_iso": lit_types.CategoryLabel(),
    }
    examples: List[lit_types.JsonDict] = []
    for vref_path, features_path, refs_path in zip(vref_paths, features_paths, refs_paths):
        features_filename = os.path.basename(features_path)
        src_iso = default_src_iso
        if features_filename != "test.src.txt":
            src_iso = features_filename.split(".")[1]

        with open(features_path, "r", encoding="utf-8") as src_file, open(
            vref_path, "r", encoding="utf-8"
        ) as vref_file:
            ref_file_paths = glob(refs_path)
            ref_files: List[IO] = []
            try:
                for ref_file_path in ref_file_paths:
                    ref_files.append(open(ref_file_path, "r", encoding="utf-8"))
                for lines in zip(src_file, vref_file, *ref_files):
                    src_line = lines[0].strip()
                    vref_line = lines[1].strip()
                    trg_iso = default_trg_iso
                    if src_line.startswith("<2"):
                        index = src_line.index(">")
                        val = src_line[2:index]
                        if val != "qaa":
                            trg_iso = val
                    example: lit_types.JsonDict = {
                        "vref": vref_line,
                        "src_text": decode_sp(src_line),
                        "src_iso": src_iso,
                        "trg_iso": trg_iso,
                    }
                    for ref_index in range(len(ref_files)):
                        ref_line = lines[ref_index + 2].strip()
                        ref_key = "ref_text" if ref_index == 0 else f"ref_text_{ref_index}"
                        example[ref_key] = ref_line
                        if ref_key not in spec:
                            spec[ref_key] = lit_types.TextSegment()
                    examples.append(example)
            finally:
                for ref_file in ref_files:
                    ref_file.close()

    return lit_dataset.Dataset(spec, examples, description="test dataset")


def create_train_dataset(
    root_dir: str, src_langs: Dict[str, Language], trg_langs: Dict[str, Language]
) -> lit_dataset.Dataset:
    src_path = os.path.join(root_dir, "train.src.txt")
    trg_path = os.path.join(root_dir, "train.trg.txt")
    default_src_iso = next(iter(src_langs.keys()))
    default_trg_iso = next(iter(trg_langs.keys()))
    examples: List[lit_types.JsonDict] = []
    with open(src_path, "r", encoding="utf-8") as src_file, open(trg_path, "r", encoding="utf-8") as trg_file:
        for src_line, trg_line in zip(src_file, trg_file):
            src_line = src_line.strip()
            trg_line = trg_line.strip()
            src_iso = default_src_iso
            if len(src_langs) > 1:
                src_iso = "?"
            trg_iso = default_trg_iso
            if src_line.startswith("<2"):
                index = src_line.index(">")
                val = src_line[2:index]
                if val != "qaa":
                    trg_iso = val
            example: lit_types.JsonDict = {
                "vref": "?",
                "src_text": decode_sp(src_line),
                "ref_text": decode_sp(trg_line),
                "src_iso": src_iso,
                "trg_iso": trg_iso,
            }
            examples.append(example)
            if len(examples) == 2000:
                break
    spec: lit_types.JsonDict = {
        "vref": lit_types.CategoryLabel(),
        "src_text": lit_types.TextSegment(),
        "ref_text": lit_types.TextSegment(),
        "src_iso": lit_types.CategoryLabel(),
        "trg_iso": lit_types.CategoryLabel(),
    }
    return lit_dataset.Dataset(spec, examples, description="train dataset")


def masked_token_mean(vectors: tf.Tensor, masks: tf.Tensor) -> tf.Tensor:
    """Mean over tokens.
    Args:
        vectors: <tf.float32>[batch_size, num_tokens, emb_dim]
        masks: <tf.bool>[batch_size, num_tokens]
    Returns:
        <tf.float32>[batch_size, emb_dim]
    """
    masks = tf.cast(masks, tf.float32)
    weights = masks / tf.reduce_sum(masks, axis=1, keepdims=True)
    return tf.reduce_sum(vectors * tf.expand_dims(weights, axis=-1), axis=1)


class NMTModel(lit_model.Model):
    def __init__(
        self,
        config: dict,
        model: SILTransformer,
        src_spp: sp.SentencePieceProcessor,
        trg_spp: sp.SentencePieceProcessor,
        step: int,
        checkpoint_path: str,
        type: str = None,
    ):
        self.types: List[str] = []
        if type is not None:
            self.types.append(type)
        self.step = step
        # Configuration priority: user config > auto config > default config.
        new_config = copy.deepcopy(_CONFIG_FALLBACK)
        merge_dict(new_config, model.auto_config())
        merge_dict(new_config, config)
        new_config["params"].setdefault("num_hypotheses", new_config["infer"].get("n_best", 1))
        new_config["params"].setdefault("average_loss_in_time", new_config["train"]["batch_type"] == "tokens")
        new_config["infer"]["n_best"] = 1
        self.config = new_config

        self.src_spp = src_spp
        self.trg_spp = trg_spp
        self.model: SILTransformer = clone_layer(model)
        self.model.initialize(self.config["data"], params=self.config["params"])
        self._analyze_fn: Optional[Function] = None

        self.checkpoint_path = checkpoint_path
        self.checkpoint: Checkpoint = None

    def description(self) -> str:
        if self.step == -1:
            return "averaged checkpoint"
        desc = f"checkpoint {self.step}"
        if len(self.types) > 0:
            types_str = ", ".join(self.types)
            desc += f" ({types_str})"
        return desc

    def predict(self, inputs: Iterable[lit_types.JsonDict], scrub_arrays=True) -> Iterator[lit_types.JsonDict]:
        inputs_list: List[str] = list(inputs)
        if len(inputs_list) == 0:
            return iter([])

        if self.checkpoint is None:
            self.checkpoint = Checkpoint.from_config(self.config, self.model)
            self.checkpoint.restore(checkpoint_path=self.checkpoint_path, weights_only=True)

        predictions: Iterator[lit_types.JsonDict] = iter(self._analyze(inputs_list))
        if scrub_arrays:
            predictions = (lit_model.scrub_numpy_refs(res) for res in predictions)
        return predictions

    def predict_single(self, one_input: lit_types.JsonDict, **kw) -> lit_types.JsonDict:
        return next(self.predict([one_input], **kw))

    def _analyze(self, inputs_list: List[lit_types.JsonDict]) -> List[lit_types.JsonDict]:
        features_list: List[str] = list(map(lambda input: encode_sp(self.src_spp, input["src_text"]), inputs_list))
        infer_config: dict = self.config["infer"]
        dataset = make_inference_dataset(
            self.model,
            features_list,
            infer_config["batch_size"],
            length_bucket_width=infer_config["length_bucket_width"],
            prefetch_buffer_size=infer_config.get("prefetch_buffer_size"),
        )

        if self._analyze_fn is None:
            self._analyze_fn = tf.function(self.model.analyze, input_signature=(dataset.element_spec,))
            if not tf.config.functions_run_eagerly():
                tf.get_logger().info("Tracing and optimizing the analyze graph...")
                self._analyze_fn.get_concrete_function()  # Trace the function now.

        results: List[lit_types.JsonDict] = [None] * len(features_list)
        for features in dataset:
            predictions = self._analyze_fn(features)

            top_k_probs, top_k_ids = tf.nn.top_k(tf.nn.softmax(predictions["logits"]), k=10)
            del predictions["logits"]
            predictions["top_k_probs"] = top_k_probs
            predictions["top_k_ids"] = top_k_ids

            masks = tf.sequence_mask(features["length"], maxlen=tf.shape(features["ids"])[1])
            predictions["encoder_final_embedding"] = masked_token_mean(predictions["encoder_outputs"], masks)
            del predictions["encoder_outputs"]

            predictions = tf.nest.map_structure(lambda t: t.numpy(), predictions)
            for prediction in extract_batches(predictions):
                index: int = prediction["index"]
                target_length = prediction["length"]
                trg_tokens = prediction["tokens"][:target_length]
                tok_trg_text = self.model.labels_inputter.tokenizer.detokenize(trg_tokens)
                trg_text = decode_sp(tok_trg_text)
                attention = prediction["alignment"][:target_length]
                probs = prediction["top_k_probs"]
                ids = prediction["top_k_ids"]
                pred_tokens = list(self._convert_top_k(ids, probs, target_length))
                encoder_final_embedding = prediction["encoder_final_embedding"]
                ref_text = inputs_list[index]["ref_text"]
                tok_ref_text = encode_sp(self.trg_spp, ref_text)
                ter_score = sacrebleu.sentence_ter(tok_trg_text, [tok_ref_text])
                chrf_score = sacrebleu.sentence_chrf(trg_text, [ref_text], order=3)
                results[index] = {
                    "trg_tokens": list(map(lambda t: t.decode("utf-8"), trg_tokens)),
                    "trg_text": trg_text,
                    "attention": np.expand_dims(attention, axis=0),
                    "src_tokens": features_list[index].split(),
                    "pred_tokens": pred_tokens,
                    "encoder_final_embedding": encoder_final_embedding,
                    "ter": ter_score.score,
                    "chrf3": chrf_score.score,
                }
        return results

    def _convert_top_k(
        self, top_k_ids: tf.Tensor, top_k_probs: tf.Tensor, trg_len: int
    ) -> Iterator[List[Tuple[str, float]]]:
        count = 0
        for token_ids, token_probs in zip(top_k_ids, top_k_probs):
            tokens = self.model.labels_inputter.ids_to_tokens.lookup(tf.cast(token_ids, tf.int64)).numpy()
            yield list(zip(map(lambda t: t.decode("utf-8"), tokens), token_probs))
            count += 1
            if count == trg_len:
                break

    def predict_minibatch(self, inputs: List[lit_types.JsonDict], config) -> List[lit_types.JsonDict]:
        pass

    def input_spec(self) -> lit_types.Spec:
        return {"src_text": lit_types.TextSegment(), "ref_text": lit_types.TextSegment()}

    def output_spec(self) -> lit_types.Spec:
        return {
            "src_tokens": lit_types.Tokens(parent="src_text"),
            "trg_text": lit_types.GeneratedText(parent="ref_text"),
            "trg_tokens": lit_types.Tokens(parent="trg_text"),
            "attention": lit_types.AttentionHeads(align_in="src_tokens", align_out="trg_tokens"),
            "pred_tokens": lit_types.TokenTopKPreds(align="trg_tokens", parent="trg_text"),
            "encoder_final_embedding": lit_types.Embeddings(),
            "ter": lit_types.Scalar(),
            "chrf3": lit_types.Scalar(),
        }


class IndexerEx(index.Indexer):
    def _get_index_path(self, index_key: str) -> str:
        return super()._get_index_path(index_key.replace(":", "-"))

    def _get_lookup_path(self, lookup_key: str) -> str:
        return super()._get_lookup_path(lookup_key.replace(":", "-"))


class BLEUMetrics(metrics.SimpleMetrics):
    def __init__(self, data_config: dict):
        self._data_config = data_config

    def is_compatible(self, field_spec: lit_types.LitType) -> bool:
        return isinstance(field_spec, lit_types.GeneratedText)

    def compute(
        self,
        labels: Sequence[Text],
        preds: Sequence[Text],
        label_spec: lit_types.TextSegment,
        pred_spec: lit_types.GeneratedText,
        config: Optional[lit_types.JsonDict] = None,
    ) -> Dict[Text, float]:
        del label_spec
        del pred_spec
        del config

        if not labels or not preds:
            return {}

        bleu_score = sacrebleu.corpus_bleu(
            preds, [labels], lowercase=True, tokenize=self._data_config.get("sacrebleu_tokenize", "13a"),
        )
        return {"bleu": bleu_score.score}


def create_lit_args(exp_name: str, checkpoints: Set[str] = {"last"}) -> Tuple[tuple, dict]:
    root_dir = get_mt_root_dir(exp_name)
    config = load_config(exp_name)
    model_dir: str = config["model_dir"]
    data_config: dict = config["data"]
    src_langs = parse_langs(data_config["src_langs"])
    trg_langs = parse_langs(data_config["trg_langs"])

    datasets: Dict[str, lit_dataset.Dataset] = {"test": create_test_dataset(root_dir, src_langs, trg_langs)}

    if data_config["share_vocab"]:
        model_prefix = os.path.join(root_dir, "sp")
        src_spp = sp.SentencePieceProcessor()
        src_spp.Load(f"{model_prefix}.model")

        trg_spp = src_spp
    else:
        src_spp = sp.SentencePieceProcessor()
        src_spp.Load(os.path.join(root_dir, "src-sp.model"))

        trg_spp = sp.SentencePieceProcessor()
        trg_spp.Load(os.path.join(root_dir, "trg-sp.model"))

    model = create_model(config)

    models: Dict[str, NMTModel] = {}

    for checkpoint in checkpoints:
        if checkpoint == "avg":
            checkpoint_path, _ = get_last_checkpoint(os.path.join(model_dir, "avg"))
            models["avg"] = NMTModel(config, model, src_spp, trg_spp, -1, checkpoint_path, type="avg")
        elif checkpoint == "last":
            last_checkpoint_path, last_step = get_last_checkpoint(model_dir)
            step_str = str(last_step)
            if step_str in models:
                models[step_str].types.append("last")
            else:
                models[str(last_step)] = NMTModel(
                    config, model, src_spp, trg_spp, last_step, last_checkpoint_path, type="last"
                )
        elif checkpoint == "best":
            best_model_path, best_step = get_best_model_dir(model_dir)
            step_str = str(best_step)
            if step_str in models:
                models[step_str].types.append("best")
            else:
                models[step_str] = NMTModel(
                    config, model, src_spp, trg_spp, best_step, os.path.join(best_model_path, "ckpt"), type="best"
                )
        else:
            checkpoint_path = os.path.join(model_dir, f"ckpt-{checkpoint}")
            step = int(checkpoint)
            models[checkpoint] = NMTModel(config, model, src_spp, trg_spp, step, checkpoint_path)

    index_datasets: Dict[str, lit_dataset.Dataset] = {"test": create_train_dataset(root_dir, src_langs, trg_langs)}
    indexer = IndexerEx(
        models,
        lit_dataset.IndexedDataset.index_all(index_datasets, caching.input_hash),
        data_dir=os.path.join(root_dir, "lit-index"),
        initialize_new_indices=True,
    )

    generators: Dict[str, lit_components.Generator] = {
        "word_replacer": word_replacer.WordReplacer(),
        "similarity_searcher": similarity_searcher.SimilaritySearcher(indexer),
    }

    interpreters: Dict[str, lit_components.Interpreter] = {
        "metrics": lit_components.ComponentGroup({"bleu": BLEUMetrics(data_config)}),
        "pca": projection.ProjectionManager(pca.PCAModel),
        "umap": projection.ProjectionManager(umap.UmapModel),
    }

    return ((models, datasets), {"generators": generators, "interpreters": interpreters})


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyzes an NMT model using LIT")
    parser.add_argument("experiment", help="Experiment name")
    parser.add_argument("--memory-growth", default=False, action="store_true", help="Enable memory growth")
    parser.add_argument(
        "--eager-execution", default=False, action="store_true", help="Enable TensorFlow eager execution.",
    )
    parser.add_argument("--checkpoint", type=str, help="Analyze checkpoint")
    parser.add_argument("--last", default=False, action="store_true", help="Analyze last checkpoint")
    parser.add_argument("--best", default=False, action="store_true", help="Analyze best evaluated checkpoint")
    parser.add_argument("--avg", default=False, action="store_true", help="Analyze averaged checkpoint")
    args = parser.parse_args()

    print("Git commit:", get_git_revision_hash())

    set_log_level(logging.INFO)

    if args.eager_execution:
        tf.config.run_functions_eagerly(True)

    if args.memory_growth:
        gpus = tf.config.list_physical_devices(device_type="GPU")
        for device in gpus:
            tf.config.experimental.set_memory_growth(device, enable=True)

    checkpoints: Set[str] = set()
    if args.avg:
        checkpoints.add("avg")
    if args.checkpoint is not None:
        checkpoints.add(args.checkpoint)
    if args.last:
        checkpoints.add("last")
    if args.best:
        checkpoints.add("best")
    if len(checkpoints) == 0:
        checkpoints.add("last")

    server_args, server_kw = create_lit_args(args.experiment, checkpoints)

    lit_server = dev_server.Server(*server_args, **server_kw, **server_flags.get_flags())
    lit_server.serve()


if __name__ == "__main__":
    main()
