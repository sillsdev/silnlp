import logging
from typing import Any, List, Optional, Tuple

import numpy as np
import tensorflow as tf
import yaml
from opennmt import Runner, evaluation
from opennmt.data import Vocab, inference_pipeline
from opennmt.models import Model, SequenceToSequence
from opennmt.utils import checkpoint as checkpoint_util
from opennmt.utils.checkpoint import Checkpoint
from opennmt.utils.misc import OrderRestorer, extract_batches, item_or_tuple


class VariableUpdate:
    def __init__(
        self, ref_variable: tf.Variable, new_variable: tf.Variable, vocab_axis: int = 0, initial: np.ndarray = None
    ) -> None:
        self.ref_variable = ref_variable
        self.new_variable = new_variable
        self.vocab_axis = vocab_axis
        self.initial = initial


def load_vec(vec_path: str) -> Tuple[List[str], np.ndarray]:
    """
    Load vocabulary and weight matrix from .vec file.
    """
    with open(vec_path, "r") as vec:
        embed_dim = int(vec.readline().split()[1])
        vocab: Any = np.loadtxt(vec_path, dtype=str, comments=None, skiprows=1, usecols=0, encoding="utf-8").tolist()
        weight = np.loadtxt(
            vec_path, dtype=np.float32, comments=None, skiprows=1, usecols=range(1, embed_dim + 1), encoding="utf-8"
        )
    return vocab, weight


def get_special_token_mapping(vocab_path: str, new_vocab_path: str) -> List[int]:
    vocab = Vocab.from_file(vocab_path)
    new_vocab = Vocab.from_file(new_vocab_path)
    mapping = [-1] * new_vocab.size
    mapping[0] = 0
    mapping[1] = 1
    mapping[2] = 2
    mapping.append(vocab.size)
    return mapping


def update_variable(update: VariableUpdate, mapping: List[int]) -> tf.Variable:
    """
    Update a vocabulary variable, possibly copying previous entries based on mapping.
    """
    ref = update.ref_variable.numpy()
    new = update.initial
    if new is None:
        new = np.zeros(update.new_variable.shape.as_list(), dtype=update.new_variable.dtype.as_numpy_dtype)
    assert (
        list(new.shape) == update.new_variable.shape.as_list()
    ), "Initial weights shape does not match expected shape."
    assert new.dtype == update.new_variable.dtype.as_numpy_dtype, "Initial weights type does not match expected type."
    perm = None
    if update.vocab_axis != 0:
        # Make the dimension to index the first.
        perm = list(range(len(ref.shape)))
        perm[0], perm[update.vocab_axis] = perm[update.vocab_axis], perm[0]
        ref = np.transpose(ref, axes=perm)
        new = np.transpose(new, axes=perm)
    for i, j in enumerate(mapping):
        if j >= 0:
            new[i] = ref[j]
    if perm is not None:
        new = np.transpose(new, axes=perm)
    update.new_variable.assign(new)
    return update.new_variable


def update_variable_and_slots(
    update: VariableUpdate,
    mapping: List[int],
    ref_optimizer: tf.keras.optimizers.Optimizer,
    new_optimizer: tf.keras.optimizers.Optimizer,
) -> List[tf.Variable]:
    """Update a vocabulary variable and its associated optimizer slots (if any)."""
    variables = [update_variable(update, mapping)]
    ref_slot_names = ref_optimizer.get_slot_names()
    new_slot_names = new_optimizer.get_slot_names()
    for slot_name in ref_slot_names:
        if slot_name not in new_slot_names:
            continue
        ref_slot = ref_optimizer.get_slot(update.ref_variable, slot_name)
        new_slot = new_optimizer.get_slot(update.new_variable, slot_name)
        slot_update = VariableUpdate(ref_slot, new_slot, update.vocab_axis)
        variables.append(update_variable(slot_update, mapping))
    return variables


def map_variables(
    updates: List[VariableUpdate],
    mapping: List[int],
    optimizer: tf.keras.optimizers.Optimizer,
    new_optimizer: tf.keras.optimizers.Optimizer,
) -> List[tf.Variable]:
    updated_variables: List[tf.Variable] = []
    for update in updates:
        variables = update_variable_and_slots(update, mapping, optimizer, new_optimizer)
        updated_variables.extend(variables)
    return updated_variables


def transfer_weights(
    model: Model,
    new_model: Model,
    optimizer: tf.keras.optimizers.Optimizer,
    new_optimizer: tf.keras.optimizers.Optimizer,
    ignore_weights: list = None,
) -> None:
    if type(model) is not type(new_model):
        raise ValueError("Transferring weights to another model type is not supported")
    if ignore_weights is None:
        ignore_weights = list()
    ignore_weights_ref = set(weight.experimental_ref() for weight in ignore_weights)
    weights = model.weights
    new_weights = new_model.weights
    for weight, new_weight in zip(weights, new_weights):
        if new_weight.experimental_ref() not in ignore_weights_ref:
            new_weight.assign(weight)
            for slot_name in new_optimizer.get_slot_names():
                if slot_name not in optimizer.get_slot_names():
                    continue
                new_slot = new_optimizer.get_slot(new_weight, slot_name)
                slot = optimizer.get_slot(weight, slot_name)
                new_slot.assign(slot)


def make_inference_dataset(
    model: Model,
    features_list: List[str],
    batch_size: int,
    batch_type: str = "examples",
    length_bucket_width: int = None,
    num_threads: int = 1,
    prefetch_buffer_size: int = None,
):
    def _map_fn(*arg):
        features = model.features_inputter.make_features(element=item_or_tuple(arg), training=False)
        if isinstance(features, (list, tuple)):
            # Special case for unsupervised inputters that always return a
            # tuple (features, labels).
            return features[0]
        return features

    transform_fns = [lambda dataset: dataset.map(_map_fn, num_parallel_calls=num_threads or 1)]

    dataset = tf.data.Dataset.from_tensor_slices(features_list)
    dataset = dataset.apply(
        inference_pipeline(
            batch_size,
            batch_type=batch_type,
            transform_fns=transform_fns,
            length_bucket_width=length_bucket_width,
            length_fn=model.features_inputter.get_length,
            num_threads=num_threads,
            prefetch_buffer_size=prefetch_buffer_size,
        )
    )
    return dataset


class SILRunner(Runner):
    def _finalize_config(self, training=False, num_replicas=1, num_devices=1):
        config = super()._finalize_config(training, num_replicas, num_devices)
        if training and not config["eval"].get("use_dictionary", True):
            data_config = config["data"]
            if "source_dictionary" in data_config:
                del data_config["source_dictionary"]
            if "target_dictionary" in data_config:
                del data_config["target_dictionary"]
        return config

    def export_embeddings(self, side: str, output_path: str) -> None:
        config = self._finalize_config()
        model = self._init_model(config)
        optimizer = model.get_optimizer()
        cur_checkpoint = Checkpoint.from_config(config, model, optimizer=optimizer)
        cur_checkpoint.restore()
        model.create_variables(optimizer=optimizer)
        vocab_file: str
        if side == "source":
            vocab_file = model.features_inputter.vocabulary_file
            embeddings_var = model.features_inputter.embedding
        else:
            vocab_file = model.labels_inputter.vocabulary_file
            embeddings_var = model.labels_inputter.embedding
        vocab = Vocab.from_file(vocab_file)
        embeddings = embeddings_var.numpy()

        with open(output_path, "w", encoding="utf-8") as file:
            file.write(f"{embeddings.shape[0] - 1} {embeddings.shape[1]}\n")
            for word, embedding in zip(vocab.words, embeddings):
                file.write(word)
                for val in embedding:
                    file.write(f" {val:1.7f}")
                file.write("\n")

    def replace_embeddings(self, side: str, embed_path: str, output_dir: str, vocab_path: str) -> None:
        _, weight = load_vec(embed_path)
        weight = np.append(weight, np.zeros((1, weight.shape[1]), dtype=weight.dtype), axis=0)

        config = self._finalize_config()

        model = self._init_model(config)
        optimizer = model.get_optimizer()
        cur_checkpoint = Checkpoint.from_config(config, model, optimizer=optimizer)
        cur_checkpoint.restore()
        model.create_variables(optimizer=optimizer)

        self._config["model_dir"] = output_dir
        if side == "source":
            self._config["data"]["source_vocabulary"] = vocab_path
        elif side == "target":
            self._config["data"]["target_vocabulary"] = vocab_path
        new_config = self._finalize_config()
        new_model = self._init_model(new_config)
        new_optimizer = new_model.get_optimizer()
        new_checkpoint = Checkpoint.from_config(new_config, new_model, optimizer=new_optimizer)
        new_model.create_variables(optimizer=new_optimizer)

        mapping: List[int]
        updates: List[VariableUpdate]
        if side == "source":
            mapping = get_special_token_mapping(
                model.features_inputter.vocabulary_file, new_model.features_inputter.vocabulary_file
            )
            updates = [
                VariableUpdate(model.features_inputter.embedding, new_model.features_inputter.embedding, initial=weight)
            ]
        else:
            mapping = get_special_token_mapping(
                model.labels_inputter.vocabulary_file, new_model.labels_inputter.vocabulary_file
            )
            updates = [
                VariableUpdate(model.labels_inputter.embedding, new_model.labels_inputter.embedding, initial=weight),
                VariableUpdate(model.decoder.output_layer.kernel, new_model.decoder.output_layer.kernel, 1),
                VariableUpdate(model.decoder.output_layer.bias, new_model.decoder.output_layer.bias),
            ]

        updated_variables = map_variables(updates, mapping, optimizer, new_optimizer)
        transfer_weights(model, new_model, optimizer, new_optimizer, ignore_weights=updated_variables)
        new_optimizer.iterations.assign(optimizer.iterations)
        new_checkpoint.save()

    def update_vocab(
        self,
        output_dir: str,
        src_vocab: Optional[str] = None,
        tgt_vocab: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        step: int = None,
    ) -> str:
        if not isinstance(self._model, SequenceToSequence):
            raise ValueError("Updating vocabularies is only supported for sequence to sequence models")
        config: dict = self._finalize_config()
        if src_vocab is None and tgt_vocab is None:
            return config["model_dir"]

        model: Model = self._init_model(config)
        optimizer = model.get_optimizer()
        cur_checkpoint = Checkpoint.from_config(config, model, optimizer=optimizer)
        cur_checkpoint.restore(checkpoint_path=checkpoint_path)
        model.create_variables(optimizer=optimizer)

        self._config["model_dir"] = output_dir
        if src_vocab is not None:
            self._config["data"]["source_vocabulary"] = src_vocab
        if tgt_vocab is not None:
            self._config["data"]["target_vocabulary"] = tgt_vocab
        new_config: dict = self._finalize_config()
        new_model: Model = self._init_model(new_config)
        new_optimizer = new_model.get_optimizer()
        new_checkpoint = Checkpoint.from_config(new_config, new_model, optimizer=new_optimizer)
        new_model.create_variables(optimizer=new_optimizer)

        model.transfer_weights(new_model, new_optimizer=new_optimizer, optimizer=optimizer)
        new_optimizer.iterations.assign(optimizer.iterations)
        new_checkpoint.save(step=step)
        return output_dir

    def infer_list(self, features_list: List[str], checkpoint_path: Optional[str] = None) -> List[List[str]]:
        config = self._finalize_config()
        model: Model = self._init_model(config)
        checkpoint = Checkpoint.from_config(config, model)
        checkpoint.restore(checkpoint_path=checkpoint_path, weights_only=True)
        infer_config = config["infer"]
        dataset = make_inference_dataset(
            model,
            features_list,
            infer_config["batch_size"],
            length_bucket_width=infer_config["length_bucket_width"],
            prefetch_buffer_size=infer_config.get("prefetch_buffer_size"),
        )

        infer_fn = tf.function(model.infer, input_signature=(dataset.element_spec,))
        if not tf.config.functions_run_eagerly():
            tf.get_logger().info("Tracing and optimizing the inference graph...")
            infer_fn.get_concrete_function()  # Trace the function now.

        results: List[List[str]] = [[""]] * len(features_list)
        for source in dataset:
            predictions = infer_fn(source)
            predictions = tf.nest.map_structure(lambda t: t.numpy(), predictions)
            for prediction in extract_batches(predictions):
                index: int = prediction["index"]
                num_hypotheses = len(prediction["log_probs"])
                hypotheses: List[str] = []
                for i in range(num_hypotheses):
                    if "tokens" in prediction:
                        target_length = prediction["length"][i]
                        tokens = prediction["tokens"][i][:target_length]
                        sentence = model.labels_inputter.tokenizer.detokenize(tokens)
                    else:
                        sentence = prediction["text"][i]
                    hypotheses.append(sentence)
                results[index] = hypotheses
        return results

    def infer_multiple(
        self, features_paths: List[str], predictions_paths: List[str], checkpoint_path: Optional[str] = None
    ) -> None:
        config = self._finalize_config()
        model: Model = self._init_model(config)
        checkpoint = Checkpoint.from_config(config, model)
        checkpoint.restore(checkpoint_path=checkpoint_path, weights_only=True)
        infer_config = config["infer"]
        for features_path, predictions_path in zip(features_paths, predictions_paths):
            dataset = model.examples_inputter.make_inference_dataset(
                features_path,
                infer_config["batch_size"],
                length_bucket_width=infer_config["length_bucket_width"],
                prefetch_buffer_size=infer_config.get("prefetch_buffer_size"),
            )

            with open(predictions_path, encoding="utf-8", mode="w") as stream:
                infer_fn = tf.function(model.infer, input_signature=(dataset.element_spec,))
                if not tf.config.functions_run_eagerly():
                    tf.get_logger().info("Tracing and optimizing the inference graph...")
                    infer_fn.get_concrete_function()  # Trace the function now.

                # Inference might return out-of-order predictions. The OrderRestorer utility is
                # used to write predictions in their original order.
                ordered_writer = OrderRestorer(
                    lambda pred: pred.get("index"),
                    lambda pred: (model.print_prediction(pred, params=infer_config, stream=stream)),
                )

                for source in dataset:
                    predictions = infer_fn(source)
                    predictions = tf.nest.map_structure(lambda t: t.numpy(), predictions)
                    for prediction in extract_batches(predictions):
                        ordered_writer.push(prediction)

    def save_effective_config(self, path: str, training: bool = False) -> None:
        level = tf.get_logger().level
        tf.get_logger().setLevel(logging.WARN)
        config = self._finalize_config(training=training)
        tf.get_logger().setLevel(level)
        with open(path, "w") as file:
            yaml.dump(config, file)

    def evaluate(self, features_file=None, labels_file=None, checkpoint_path=None):
        """Runs evaluation.

        Args:
          features_file: The input features file to evaluate. If not set, will load
            ``eval_features_file`` from the data configuration.
          labels_file: The output labels file to evaluate. If not set, will load
            ``eval_labels_file`` from the data configuration.
          checkpoint_path: The checkpoint path to load the model weights from.

        Returns:
          A dict of evaluation metrics.
        """
        config = self._finalize_config()
        model = self._init_model(config)
        checkpoint = checkpoint_util.Checkpoint.from_config(config, model)
        checkpoint_path = checkpoint.restore(checkpoint_path=checkpoint_path, weights_only=True)
        step = checkpoint_util.get_step_from_checkpoint_prefix(checkpoint_path)
        evaluator = evaluation.Evaluator.from_config(
            model, config, features_file=features_file, labels_file=labels_file
        )

        results = evaluator(step)

        for key, value in results.items():
            tf.summary.scalar(key, value, step=step)

        return results