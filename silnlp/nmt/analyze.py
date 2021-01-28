import argparse
import copy
import logging
import os
from glob import glob
from typing import IO, Dict, Iterable, Iterator, List, Tuple

logging.basicConfig()

import numpy as np
import opennmt.data
import opennmt.models
import opennmt.runner
import opennmt.utils.checkpoint
import opennmt.utils.misc
import sentencepiece as sp
import tensorflow as tf
from lit_nlp import dev_server, server_flags
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types

from ..common.utils import get_git_revision_hash, get_mt_root_dir
from .config import Language, create_model, load_config, parse_langs, set_log_level
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
        "src": lit_types.TextSegment(),
        "src_iso": lit_types.CategoryLabel(),
        "trg_iso": lit_types.CategoryLabel(),
        "vref": lit_types.CategoryLabel(),
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
                        "src": decode_sp(src_line),
                        "src_iso": src_iso,
                        "trg_iso": trg_iso,
                        "vref": vref_line,
                    }
                    for ref_index in range(len(ref_files)):
                        ref_line = lines[ref_index + 2].strip()
                        ref_key = "ref" if ref_index == 0 else f"ref_{ref_index}"
                        example[ref_key] = ref_line
                        if ref_key not in spec:
                            spec[ref_key] = lit_types.TextSegment()
                    examples.append(example)
            finally:
                for ref_file in ref_files:
                    ref_file.close()

    return lit_dataset.Dataset(spec, examples, description="test dataset")


class NMTModel(lit_model.Model):
    def __init__(
        self,
        config: dict,
        model: opennmt.models.Model,
        src_spp: sp.SentencePieceProcessor,
        trg_spp: sp.SentencePieceProcessor,
        type: str,
        step: int,
        checkpoint_path: str,
    ):
        self.type = type
        self.step = step
        # Configuration priority: user config > auto config > default config.
        new_config = copy.deepcopy(opennmt.runner._CONFIG_FALLBACK)
        opennmt.utils.misc.merge_dict(new_config, model.auto_config())
        opennmt.utils.misc.merge_dict(new_config, config)
        new_config["params"].setdefault("num_hypotheses", new_config["infer"].get("n_best", 1))
        new_config["params"].setdefault("average_loss_in_time", new_config["train"]["batch_type"] == "tokens")
        self.config = new_config

        self.src_spp = src_spp
        self.trg_spp = trg_spp
        self.model = opennmt.utils.misc.clone_layer(model)
        self.model.initialize(self.config["data"], params=self.config["params"])

        self.checkpoint_path = checkpoint_path
        self.checkpoint: opennmt.utils.checkpoint.Checkpoint = None

    def description(self) -> str:
        return f"{self.type} checkpoint ({self.step})"

    def predict(self, inputs: Iterable[lit_types.JsonDict], scrub_arrays=True) -> Iterator[lit_types.JsonDict]:
        features_list: List[str] = list(map(lambda input: encode_sp(self.src_spp, input["src"]), inputs))
        if len(features_list) == 0:
            return []

        if self.checkpoint is None:
            self.checkpoint = opennmt.utils.checkpoint.Checkpoint.from_config(self.config, self.model)
            self.checkpoint.restore(checkpoint_path=self.checkpoint_path, weights_only=True)

        predictions = self._infer(features_list)

        labels_list: List[str] = list(map(lambda p: " ".join(p["trg_tokens"]), predictions))
        top_k_list = self._get_top_k_tokens(features_list, labels_list)

        for features, top_k, prediction in zip(features_list, top_k_list, predictions):
            prediction["src_tokens"] = features.split()
            opennmt.utils.misc.merge_dict(prediction, top_k)

        if scrub_arrays:
            predictions = (lit_model.scrub_numpy_refs(res) for res in predictions)
        return predictions

    def predict_single(self, one_input: lit_types.JsonDict, **kw) -> lit_types.JsonDict:
        return list(self.predict([one_input], **kw))[0]

    def _infer(self, features_list: List[str]) -> List[lit_types.JsonDict]:
        infer_config: dict = self.config["infer"]
        dataset = self._make_inference_dataset(
            features_list,
            infer_config["batch_size"],
            length_bucket_width=infer_config["length_bucket_width"],
            prefetch_buffer_size=infer_config.get("prefetch_buffer_size"),
        )

        infer_fn = tf.function(self.model.infer, input_signature=(dataset.element_spec,))
        if not tf.config.functions_run_eagerly():
            tf.get_logger().info("Tracing and optimizing the inference graph...")
            infer_fn.get_concrete_function()  # Trace the function now.

        results: List[lit_types.JsonDict] = [None] * len(features_list)
        for features in dataset:
            predictions = infer_fn(features)
            predictions = tf.nest.map_structure(lambda t: t.numpy(), predictions)
            for prediction in opennmt.utils.misc.extract_batches(predictions):
                target_length = prediction["length"][0]
                tokens = prediction["tokens"][0][:target_length]
                sentence = self.model.examples_inputter.labels_inputter.tokenizer.detokenize(tokens)
                attention = prediction["alignment"][0][:target_length]
                attention = np.delete(attention, attention.shape[1] - 1, axis=1)
                results[prediction["index"]] = {
                    "trg_tokens": list(map(lambda t: t.decode("utf-8"), tokens)),
                    "trg": decode_sp(sentence),
                    "attention": np.expand_dims(attention, axis=0),
                }
        return results

    def _get_top_k_tokens(self, features_list: List[str], labels_list: List[str]) -> List[lit_types.JsonDict]:
        infer_config: dict = self.config["infer"]
        dataset = self._make_evaluation_dataset(
            features_list,
            labels_list,
            infer_config["batch_size"],
            length_bucket_width=infer_config["length_bucket_width"],
            prefetch_buffer_size=infer_config.get("prefetch_buffer_size"),
        )

        call_fn = tf.function(self.model.call, input_signature=dataset.element_spec)
        if not tf.config.functions_run_eagerly():
            tf.get_logger().info("Tracing and optimizing the call graph...")
            call_fn.get_concrete_function()  # Trace the function now.

        results: List[lit_types.JsonDict] = [None] * len(features_list)
        for features, labels in dataset:
            outputs, _ = call_fn(features, labels)
            outputs["length"] = labels["length"]
            outputs["ids_out"] = labels["ids_out"]
            top_k_probs, top_k_ids = tf.nn.top_k(tf.nn.softmax(outputs["logits"]), k=10)
            outputs["top_k_probs"] = top_k_probs
            outputs["top_k_ids"] = top_k_ids
            outputs["index"] = features["index"]
            outputs = tf.nest.map_structure(lambda t: t.numpy(), outputs)
            for output in opennmt.utils.misc.extract_batches(outputs):
                index = output["index"]
                trg_len = output["length"] - 1
                probs = output["top_k_probs"]
                ids = output["top_k_ids"]
                top_k = list(self._convert_top_k(ids, probs, trg_len))
                results[index] = {"top_k_tokens": top_k}

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

    def _make_inference_dataset(
        self,
        features_list: List[str],
        batch_size: int,
        batch_type: str = "examples",
        length_bucket_width: int = None,
        num_threads: int = 1,
        prefetch_buffer_size: int = None,
    ):
        def _map_fn(*arg):
            features = self.model.features_inputter.make_features(
                element=opennmt.utils.misc.item_or_tuple(arg), training=False
            )
            if isinstance(features, (list, tuple)):
                # Special case for unsupervised inputters that always return a
                # tuple (features, labels).
                return features[0]
            return features

        transform_fns = [lambda dataset: dataset.map(_map_fn, num_parallel_calls=num_threads or 1)]

        dataset = tf.data.Dataset.from_tensor_slices(features_list)
        dataset = dataset.apply(
            opennmt.data.inference_pipeline(
                batch_size,
                batch_type=batch_type,
                transform_fns=transform_fns,
                length_bucket_width=length_bucket_width,
                length_fn=self.model.features_inputter.get_length,
                num_threads=num_threads,
                prefetch_buffer_size=prefetch_buffer_size,
            )
        )
        return dataset

    def _make_evaluation_dataset(
        self,
        features_list: List[str],
        labels_list: List[str],
        batch_size,
        batch_type="examples",
        length_bucket_width=None,
        num_threads=1,
        prefetch_buffer_size=None,
    ):
        length_fn = [
            self.model.features_inputter.get_length,
            self.model.labels_inputter.get_length,
        ]
        map_fn = lambda *arg: self.model.examples_inputter.make_features(
            element=opennmt.utils.misc.item_or_tuple(arg), training=False
        )
        transform_fns = [lambda dataset: dataset.map(map_fn, num_parallel_calls=num_threads or 1)]

        dataset = tf.data.Dataset.from_tensor_slices(list(zip(features_list, labels_list)))
        dataset = dataset.apply(
            opennmt.data.inference_pipeline(
                batch_size,
                batch_type=batch_type,
                transform_fns=transform_fns,
                length_bucket_width=length_bucket_width,
                length_fn=length_fn,
                num_threads=num_threads,
                prefetch_buffer_size=prefetch_buffer_size,
            )
        )
        return dataset

    def predict_minibatch(self, inputs: List[lit_types.JsonDict], config) -> List[lit_types.JsonDict]:
        pass

    def input_spec(self) -> lit_types.Spec:
        return {"src": lit_types.TextSegment(), "ref": lit_types.TextSegment()}

    def output_spec(self) -> lit_types.Spec:
        return {
            "src_tokens": lit_types.Tokens(parent="src"),
            "trg": lit_types.GeneratedText(parent="ref"),
            "trg_tokens": lit_types.Tokens(parent="trg"),
            "attention": lit_types.AttentionHeads(align=("src_tokens", "trg_tokens")),
            "top_k_tokens": lit_types.TokenTopKPreds(align="trg_tokens", parent="trg"),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyzes an NMT model using LIT")
    parser.add_argument("experiment", help="Experiment name")
    parser.add_argument("--memory-growth", default=False, action="store_true", help="Enable memory growth")
    parser.add_argument(
        "--eager-execution", default=False, action="store_true", help="Enable TensorFlow eager execution.",
    )
    args = parser.parse_args()

    print("Git commit:", get_git_revision_hash())

    set_log_level(logging.INFO)

    if args.eager_execution:
        tf.config.run_functions_eagerly(True)

    if args.memory_growth:
        gpus = tf.config.list_physical_devices(device_type="GPU")
        for device in gpus:
            tf.config.experimental.set_memory_growth(device, enable=True)

    exp_name = args.experiment
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

    last_checkpoint_path, last_step = get_last_checkpoint(model_dir)
    best_model_path, best_step = get_best_model_dir(model_dir)
    models: Dict[str, lit_model.Model] = {
        "last": NMTModel(config, model, src_spp, trg_spp, "last", last_step, last_checkpoint_path),
        "best": NMTModel(config, model, src_spp, trg_spp, "best", best_step, os.path.join(best_model_path, "ckpt")),
    }

    lit_server = dev_server.Server(models, datasets, **server_flags.get_flags())
    lit_server.serve()


if __name__ == "__main__":
    main()
