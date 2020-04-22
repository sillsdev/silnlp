from typing import List, Tuple

import numpy as np
import opennmt
import tensorflow as tf


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
        vocab = np.loadtxt(vec_path, dtype=str, comments=None, skiprows=1, usecols=0, encoding="utf-8").tolist()
        weight = np.loadtxt(
            vec_path, dtype=np.float32, comments=None, skiprows=1, usecols=range(1, embed_dim + 1), encoding="utf-8"
        )
    return vocab, weight


def get_special_token_mapping(vocab_path: str, new_vocab_path: str) -> List[int]:
    vocab = opennmt.data.Vocab.from_file(vocab_path)
    new_vocab = opennmt.data.Vocab.from_file(new_vocab_path)
    mapping = [-1] * new_vocab.size
    mapping[0] = 0
    mapping[1] = 1
    mapping[2] = 2
    mapping.append(vocab.size)
    return mapping


def update_variable(update: VariableUpdate, mapping: List[int]) -> tf.Variable:
    """Update a vocabulary variable, possibly copying previous entries based on
  mapping.
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
    variables = []
    variables.append(update_variable(update, mapping))
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
    model: opennmt.models.Model,
    new_model: opennmt.models.Model,
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


class RunnerEx(opennmt.Runner):
    def export_embeddings(self, side: str, output_path: str, checkpoint_path: str = None) -> None:
        config = self._finalize_config()
        model = self._init_model(config)
        optimizer = model.get_optimizer()
        cur_checkpoint = opennmt.utils.checkpoint.Checkpoint.from_config(config, model, optimizer=optimizer)
        cur_checkpoint.restore()
        model.create_variables(optimizer=optimizer)
        if side == "source":
            vocab_file = model.features_inputter.vocabulary_file
            embeddings_var = model.features_inputter.embedding
        elif side == "target":
            vocab_file = model.labels_inputter.vocabulary_file
            embeddings_var = model.labels_inputter.embedding
        vocab = opennmt.data.Vocab.from_file(vocab_file)
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
        cur_checkpoint = opennmt.utils.checkpoint.Checkpoint.from_config(config, model, optimizer=optimizer)
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
        new_checkpoint = opennmt.utils.checkpoint.Checkpoint.from_config(new_config, new_model, optimizer=new_optimizer)
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
        elif side == "target":
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
