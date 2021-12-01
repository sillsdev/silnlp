from typing import Any, List, Optional, Tuple

import tensorflow as tf
import tensorflow_addons as tfa
from opennmt import END_OF_SENTENCE_ID, START_OF_SENTENCE_ID
from opennmt.data.vocab import get_mapping, update_variable, update_variable_and_slots
from opennmt.decoders import SelfAttentionDecoder
from opennmt.encoders import ParallelEncoder, SelfAttentionEncoder
from opennmt.inputters import ParallelInputter, WordEmbedder, add_sequence_controls
from opennmt.layers import (
    Dense,
    FeedForwardNetwork,
    LayerNorm,
    MultiHeadAttention,
    MultiHeadAttentionReduction,
    SelfAttentionEncoderLayer,
    SinusoidalPositionEncoder,
    TransformerLayerWrapper,
    dropout,
    future_mask,
    split_heads,
)
from opennmt.layers.reducer import align_in_time
from opennmt.models import EmbeddingsSharingLevel, SequenceToSequence, Transformer, register_model_in_catalog
from opennmt.models.sequence_to_sequence import _add_noise, replace_unknown_target
from opennmt.utils.decoding import BeamSearch, DecodingStrategy, Sampler
from opennmt.utils.misc import shape_list

from .decoding import DictionaryGuidedBeamSearch, dynamic_decode
from .ref_inputter import RefInputter
from .take_first_reducer import TakeFirstReducer
from .trie import Trie

EPSILON: float = 1e-07


def clip_attention_probs(attention: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    attention = tf.clip_by_value(attention, EPSILON, 1.0 - EPSILON)
    mask = tf.cast(mask, attention.dtype)
    if mask.shape.rank == 2:
        mask = tf.expand_dims(mask, 1)  # Broadcast on time dimension.
    mask = tf.expand_dims(mask, 1)  # Broadcast on head dimension.
    attention = tf.multiply(attention, mask)
    return attention


class SILTransformerLayerWrapper(TransformerLayerWrapper):
    def __init__(self, layer, output_dropout, pre_norm=True, residual_connection=True, **kwargs):
        super(TransformerLayerWrapper, self).__init__(
            layer,
            normalize_input=pre_norm,
            normalize_output=not pre_norm,
            output_dropout=output_dropout,
            residual_connection=residual_connection,
            **kwargs,
        )


class SILSelfAttentionEncoderLayer(SelfAttentionEncoderLayer):
    def __init__(
        self,
        num_units,
        num_heads,
        ffn_inner_dim,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        ffn_activation=tf.nn.relu,
        maximum_relative_position=None,
        pre_norm=True,
        self_attention_residual_connection=True,
        **kwargs,
    ):
        super(SelfAttentionEncoderLayer, self).__init__(**kwargs)
        self.self_attention = MultiHeadAttention(
            num_heads,
            num_units,
            dropout=attention_dropout,
            maximum_relative_position=maximum_relative_position,
        )
        self.self_attention = SILTransformerLayerWrapper(
            self.self_attention, dropout, pre_norm=pre_norm, residual_connection=self_attention_residual_connection
        )
        self.ffn = FeedForwardNetwork(ffn_inner_dim, num_units, dropout=ffn_dropout, activation=ffn_activation)
        self.ffn = SILTransformerLayerWrapper(self.ffn, dropout, pre_norm=pre_norm)


class SILSelfAttentionEncoder(SelfAttentionEncoder):
    def __init__(
        self,
        num_layers,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        ffn_activation=tf.nn.relu,
        position_encoder_class=SinusoidalPositionEncoder,
        maximum_relative_position=None,
        pre_norm=True,
        drop_self_attention_residual_connections=set(),
        **kwargs,
    ):
        super(SelfAttentionEncoder, self).__init__(**kwargs)
        self.num_units = num_units
        self.dropout = dropout
        self.position_encoder = None
        if position_encoder_class is not None:
            self.position_encoder = position_encoder_class()
        self.layer_norm = LayerNorm() if pre_norm else None
        self.layers = [
            SILSelfAttentionEncoderLayer(
                num_units,
                num_heads,
                ffn_inner_dim,
                dropout=dropout,
                attention_dropout=attention_dropout,
                ffn_dropout=ffn_dropout,
                ffn_activation=ffn_activation,
                maximum_relative_position=maximum_relative_position,
                pre_norm=pre_norm,
                self_attention_residual_connection=i not in drop_self_attention_residual_connections,
            )
            for i in range(num_layers)
        ]


class AlignmentHead(tf.keras.layers.Layer):
    def __init__(self, num_units, **kwargs):
        super().__init__(**kwargs)
        self.num_units = num_units
        self.linear_queries = Dense(self.num_units)
        self.linear_keys = Dense(self.num_units)

    def call(self, inputs, memory=None, mask=None, cache=None):
        def _compute_k(x):
            keys = self.linear_keys(x)
            keys = split_heads(keys, 1)
            return keys

        # Compute queries.
        queries = self.linear_queries(inputs)
        queries = split_heads(queries, 1)
        queries *= self.num_units ** -0.5

        # Compute keys.
        if cache is not None:
            keys = tf.cond(
                tf.equal(tf.shape(cache)[2], 0),
                true_fn=lambda: _compute_k(memory),
                false_fn=lambda: cache,
            )
        else:
            keys = _compute_k(memory)

        cache = keys

        # Dot product attention.
        dot = tf.matmul(queries, keys, transpose_b=True)
        if mask is not None:
            mask = tf.cast(mask, tf.float32)
            if mask.shape.rank == 2:
                mask = tf.expand_dims(mask, 1)  # Broadcast on time dimension.
            mask = tf.expand_dims(mask, 1)  # Broadcast on head dimension.
            dot = tf.cast(
                tf.cast(dot, tf.float32) * mask + ((1.0 - mask) * tf.float32.min),
                dot.dtype,
            )
        attn = tf.cast(tf.nn.softmax(tf.cast(dot, tf.float32)), dot.dtype)
        return attn, cache


class SILSelfAttentionDecoderLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        num_units,
        num_heads,
        ffn_inner_dim,
        num_sources=1,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        ffn_activation=tf.nn.relu,
        maximum_relative_position=None,
        pre_norm=True,
        alignment_head_num_units=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.self_attention = MultiHeadAttention(
            num_heads,
            num_units,
            dropout=attention_dropout,
            maximum_relative_position=maximum_relative_position,
        )
        self.self_attention = TransformerLayerWrapper(self.self_attention, dropout, pre_norm=pre_norm)
        self.attention = []
        for _ in range(num_sources):
            attention = MultiHeadAttention(
                num_heads,
                num_units,
                dropout=attention_dropout,
                return_attention=alignment_head_num_units is None,
            )
            attention = TransformerLayerWrapper(attention, dropout, pre_norm=pre_norm)
            self.attention.append(attention)
        if alignment_head_num_units is None:
            self.alignment_head = None
        else:
            self.alignment_head = AlignmentHead(alignment_head_num_units)
            self.alignment_head = SILTransformerLayerWrapper(
                self.alignment_head, 0, pre_norm=pre_norm, residual_connection=False
            )
        self.ffn = FeedForwardNetwork(ffn_inner_dim, num_units, dropout=ffn_dropout, activation=ffn_activation)
        self.ffn = TransformerLayerWrapper(self.ffn, dropout, pre_norm=pre_norm)

    def call(
        self,
        inputs,
        mask=None,
        memory=None,
        memory_mask=None,
        cache=None,
        training=None,
    ):
        """Runs the decoder layer."""
        if cache is None:
            cache = {}

        outputs, self_kv = self.self_attention(inputs, mask=mask, cache=cache.get("self_kv"), training=training)

        attention = []
        memory_kv = []
        alignment_memory_k = None
        if memory is not None:
            memory_kv_cache = cache.get("memory_kv")
            if memory_kv_cache is None:
                memory_kv_cache = [None] * len(self.attention)
            for layer, mem, mem_mask, mem_cache in zip(self.attention, memory, memory_mask, memory_kv_cache):
                result = layer(
                    outputs,
                    memory=mem,
                    mask=mem_mask,
                    cache=mem_cache,
                    training=training,
                )
                if len(result) == 3:
                    outputs, memory_kv_i, attention_i = result
                    attention.append(attention_i)
                else:
                    outputs, memory_kv_i = result
                memory_kv.append(memory_kv_i)

            if self.alignment_head is not None:
                attention_0, alignment_memory_k = self.alignment_head(
                    outputs, memory=memory[0], mask=memory_mask[0], cache=cache.get("alignment_memory_k")
                )
                # clip the probs to ensure that model does not diverge with loss = NaN
                attention_0 = clip_attention_probs(attention_0, memory_mask[0])
                attention.append(attention_0)

        outputs = self.ffn(outputs, training=training)
        cache = dict(self_kv=self_kv, memory_kv=memory_kv)
        if self.alignment_head is not None:
            cache["alignment_memory_k"] = alignment_memory_k
        return outputs, cache, attention


class SILSelfAttentionDecoder(SelfAttentionDecoder):
    def __init__(
        self,
        num_layers,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        ffn_activation=tf.nn.relu,
        position_encoder_class=SinusoidalPositionEncoder,
        num_sources=1,
        maximum_relative_position=None,
        attention_reduction=MultiHeadAttentionReduction.FIRST_HEAD_LAST_LAYER,
        pre_norm=True,
        alignment_head_num_units=None,
        **kwargs,
    ):
        super(SelfAttentionDecoder, self).__init__(num_sources=num_sources, **kwargs)
        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout = dropout
        self.attention_reduction = attention_reduction
        self.position_encoder = None
        if position_encoder_class is not None:
            self.position_encoder = position_encoder_class()
        self.alignment_head_num_units = alignment_head_num_units
        self.layer_norm = LayerNorm() if pre_norm else None
        self.layers = [
            SILSelfAttentionDecoderLayer(
                self.num_units,
                self.num_heads,
                ffn_inner_dim,
                num_sources=num_sources,
                dropout=dropout,
                attention_dropout=attention_dropout,
                ffn_dropout=ffn_dropout,
                ffn_activation=ffn_activation,
                maximum_relative_position=maximum_relative_position,
                pre_norm=pre_norm,
                alignment_head_num_units=alignment_head_num_units,
            )
            for _ in range(num_layers)
        ]

    def _run(
        self,
        inputs,
        sequence_length=None,
        cache=None,
        memory=None,
        memory_sequence_length=None,
        step=None,
        training=None,
    ):
        # Process inputs.
        inputs *= self.num_units ** 0.5
        if self.position_encoder is not None:
            inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
        inputs = dropout(inputs, self.dropout, training=training)

        # Prepare query mask.
        mask = None
        if step is None:
            maximum_length = tf.shape(inputs)[1]
            if sequence_length is None:
                batch_size = tf.shape(inputs)[0]
                sequence_length = tf.fill([batch_size], maximum_length)
            mask = future_mask(sequence_length, maximum_length=maximum_length)

        # Prepare memory mask.
        memory_mask = None
        if memory is not None:
            if not isinstance(memory, (list, tuple)):
                memory = (memory,)
            if memory_sequence_length is not None:
                if not isinstance(memory_sequence_length, (list, tuple)):
                    memory_sequence_length = (memory_sequence_length,)
                memory_mask = [
                    tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
                    for mem, mem_length in zip(memory, memory_sequence_length)
                ]
            else:
                memory_mask = tuple(None for _ in memory)

        # Run each layer.
        new_cache = []
        attention = []
        for i, layer in enumerate(self.layers):
            inputs, layer_cache, layer_attention = layer(
                inputs,
                mask=mask,
                memory=memory,
                memory_mask=memory_mask,
                cache=cache[i] if cache is not None else None,
                training=training,
            )
            attention.append(layer_attention)
            new_cache.append(layer_cache)
        outputs = self.layer_norm(inputs) if self.layer_norm is not None else inputs

        # Convert list of shape num_layers x num_sources to num_sources x num_layers
        attention = list(map(list, zip(*attention)))
        if attention:
            attention = MultiHeadAttentionReduction.reduce(
                attention[0],  # Get attention to the first source.
                self.attention_reduction,
            )
        else:
            attention = None

        return outputs, new_cache, attention

    def dynamic_decode(
        self,
        embeddings,
        start_ids,
        end_id=END_OF_SENTENCE_ID,
        initial_state=None,
        decoding_strategy=None,
        sampler=None,
        maximum_iterations=None,
        minimum_iterations=0,
        tflite_output_size=None,
    ):
        if tflite_output_size is not None:
            input_fn = lambda ids: embeddings.tflite_call(ids)
        elif isinstance(embeddings, WordEmbedder):
            input_fn = lambda ids: embeddings({"ids": ids})
        else:
            input_fn = lambda ids: tf.nn.embedding_lookup(embeddings, ids)

        # TODO: find a better way to pass the state reorder flags.
        if hasattr(decoding_strategy, "_set_state_reorder_flags"):
            state_reorder_flags = self._get_state_reorder_flags()
            decoding_strategy._set_state_reorder_flags(state_reorder_flags)

        return dynamic_decode(
            lambda ids, step, state: self(input_fn(ids), step, state),
            start_ids,
            end_id=end_id,
            initial_state=initial_state,
            decoding_strategy=decoding_strategy,
            sampler=sampler,
            maximum_iterations=maximum_iterations,
            minimum_iterations=minimum_iterations,
            attention_history=self.support_alignment_history,
            attention_size=tf.shape(self.memory)[1] if self.support_alignment_history else None,
            tflite_output_size=tflite_output_size,
        )

    def _get_initial_state(self, batch_size, dtype, initial_state=None):
        cache = super()._get_initial_state(batch_size, dtype, initial_state)
        if self.alignment_head_num_units is not None:
            for layer_cache in cache:
                layer_cache["alignment_memory_k"] = tf.zeros(
                    [batch_size, 1, 0, self.alignment_head_num_units], dtype=dtype
                )
        return cache

    def _get_state_reorder_flags(self):
        reorder_flags = super()._get_state_reorder_flags()
        if self.alignment_head_num_units is not None:
            for flag in reorder_flags:
                flag["alignment_memory_k"] = False
        return reorder_flags


class SILTransformer(Transformer):
    def __init__(
        self,
        source_inputter=None,
        target_inputter=None,
        num_layers=6,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        ffn_activation=tf.nn.relu,
        position_encoder_class=SinusoidalPositionEncoder,
        share_embeddings=EmbeddingsSharingLevel.NONE,
        share_encoders=False,
        maximum_relative_position=None,
        attention_reduction=MultiHeadAttentionReduction.FIRST_HEAD_LAST_LAYER,
        pre_norm=True,
        drop_encoder_self_attention_residual_connections=set(),
        alignment_head_num_units=None,
    ):
        if source_inputter is None:
            source_inputter = WordEmbedder(embedding_size=num_units)
        if target_inputter is None:
            target_inputter = WordEmbedder(embedding_size=num_units)

        if isinstance(num_layers, (list, tuple)):
            num_encoder_layers, num_decoder_layers = num_layers
        else:
            num_encoder_layers, num_decoder_layers = num_layers, num_layers
        encoders = [
            SILSelfAttentionEncoder(
                num_encoder_layers,
                num_units=num_units,
                num_heads=num_heads,
                ffn_inner_dim=ffn_inner_dim,
                dropout=dropout,
                attention_dropout=attention_dropout,
                ffn_dropout=ffn_dropout,
                ffn_activation=ffn_activation,
                position_encoder_class=position_encoder_class,
                maximum_relative_position=maximum_relative_position,
                pre_norm=pre_norm,
                drop_self_attention_residual_connections=drop_encoder_self_attention_residual_connections,
            )
            for _ in range(source_inputter.num_outputs)
        ]
        if len(encoders) > 1:
            encoder = ParallelEncoder(
                encoders if not share_encoders else encoders[0],
                outputs_reducer=None,
                states_reducer=None,
            )
        else:
            encoder = encoders[0]
        decoder = SILSelfAttentionDecoder(
            num_decoder_layers,
            num_units=num_units,
            num_heads=num_heads,
            ffn_inner_dim=ffn_inner_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation,
            position_encoder_class=position_encoder_class,
            num_sources=source_inputter.num_outputs,
            maximum_relative_position=maximum_relative_position,
            attention_reduction=attention_reduction,
            pre_norm=pre_norm,
            alignment_head_num_units=alignment_head_num_units,
        )

        self._pre_norm = pre_norm
        self._num_units = num_units
        self._num_encoder_layers = num_encoder_layers
        self._num_decoder_layers = num_decoder_layers
        self._num_heads = num_heads
        self._with_relative_position = maximum_relative_position is not None
        self._is_ct2_compatible = (
            isinstance(encoder, SelfAttentionEncoder)
            and ffn_activation is tf.nn.relu
            and (
                (self._with_relative_position and position_encoder_class is None)
                or (not self._with_relative_position and position_encoder_class == SinusoidalPositionEncoder)
            )
        )
        self._dictionary: Optional[Trie] = None
        super(Transformer, self).__init__(
            source_inputter,
            target_inputter,
            encoder,
            decoder,
            share_embeddings=share_embeddings,
        )

    def initialize(self, data_config, params=None):
        super().initialize(data_config, params=params)
        src_dict_path: Optional[str] = data_config.get("source_dictionary")
        trg_dict_path: Optional[str] = data_config.get("target_dictionary")
        ref_dict_path: Optional[str] = data_config.get("ref_dictionary")
        if src_dict_path is not None and trg_dict_path is not None:
            self.labels_inputter.set_decoder_mode(enable=False, mark_start=False, mark_end=False)
            dictionary = Trie(self.features_inputter.vocabulary_size)
            with tf.io.gfile.GFile(src_dict_path) as src_dict, tf.io.gfile.GFile(
                trg_dict_path
            ) as trg_dict, tf.io.gfile.GFile(ref_dict_path) as ref_dict:
                for src_entry_str, trg_entry_str, ref_entry_str in zip(src_dict, trg_dict, ref_dict):
                    src_entry = src_entry_str.strip().split("\t")
                    src_ids = [
                        self.features_inputter.make_features(tf.constant(se.strip()))["inputter_0_ids"]
                        for se in src_entry
                    ]
                    trg_entry = trg_entry_str.strip().split("\t")
                    trg_ids = [self.labels_inputter.make_features(tf.constant(te.strip()))["ids"] for te in trg_entry]
                    refs = tf.convert_to_tensor(ref_entry_str.strip().split("\t"))
                    for src_variant_ids in src_ids:
                        dictionary.add(src_variant_ids, trg_ids, refs)
            if not dictionary.empty:
                dictionary.compile()
                self._dictionary = dictionary
            self.labels_inputter.set_decoder_mode(mark_start=True, mark_end=True)

    def analyze(self, features):
        # Encode the source.
        source_length = self.features_inputter.get_length(features)
        source_inputs = self.features_inputter(features)
        encoder_outputs, encoder_state, encoder_sequence_length = self.encoder(
            source_inputs, sequence_length=source_length
        )

        predictions = self._dynamic_decode(features, encoder_outputs, encoder_state, encoder_sequence_length)

        length = predictions["length"]
        length = tf.squeeze(length, axis=[1])
        tokens = predictions["tokens"]
        tokens = tf.squeeze(tokens, axis=[1])
        tokens = tf.where(tf.equal(tokens, "</s>"), tf.fill(tf.shape(tokens), ""), tokens)

        ids = self.labels_inputter.tokens_to_ids.lookup(tokens)
        if self.labels_inputter.mark_start or self.labels_inputter.mark_end:
            ids, length = add_sequence_controls(
                ids,
                length,
                start_id=START_OF_SENTENCE_ID if self.labels_inputter.mark_start else None,
                end_id=END_OF_SENTENCE_ID if self.labels_inputter.mark_end else None,
            )
        labels = {"ids_out": ids[:, 1:], "ids": ids[:, :-1], "length": length - 1}

        outputs = self._decode_target(labels, encoder_outputs, encoder_state, encoder_sequence_length)

        return {
            "length": tf.squeeze(predictions["length"], axis=[1]),
            "tokens": tf.squeeze(predictions["tokens"], axis=[1]),
            "alignment": tf.squeeze(predictions["alignment"], axis=[1]),
            "encoder_outputs": encoder_outputs,
            "logits": outputs["logits"],
            "index": features["index"],
        }

    def set_dropout(self, dropout: float = 0.1, attention_dropout: float = 0.1, ffn_dropout: float = 0.1) -> None:
        root_layer = self
        for layer in (root_layer,) + root_layer.submodules:
            name: str = layer.name
            if name == "self_attention_encoder":
                layer.dropout = dropout
            elif name == "self_attention_decoder":
                layer.dropout = dropout
            elif name.startswith("transformer_layer_wrapper"):
                layer.output_dropout = dropout
            elif name.startswith("multi_head_attention"):
                layer.dropout = attention_dropout
            elif name.startswith("feed_forward_network"):
                layer.dropout = ffn_dropout

    def _dynamic_decode(
        self,
        features,
        encoder_outputs,
        encoder_state,
        encoder_sequence_length,
        tflite_run=False,
    ):
        params = self.params
        batch_size = tf.shape(tf.nest.flatten(encoder_outputs)[0])[0]
        start_ids = tf.fill([batch_size], START_OF_SENTENCE_ID)
        beam_size = params.get("beam_width", 1)

        if beam_size > 1:
            # Tile encoder outputs to prepare for beam search.
            encoder_outputs = tfa.seq2seq.tile_batch(encoder_outputs, beam_size)
            encoder_sequence_length = tfa.seq2seq.tile_batch(encoder_sequence_length, beam_size)
            encoder_state = tf.nest.map_structure(
                lambda state: tfa.seq2seq.tile_batch(state, beam_size) if state is not None else None,
                encoder_state,
            )

        decoding_strategy = DecodingStrategy.from_params(params, tflite_mode=tflite_run)
        if self._dictionary is not None and isinstance(decoding_strategy, BeamSearch):
            src_ids: tf.Tensor = features["inputter_0_ids"]
            ref: tf.Tensor = features["inputter_1_ref"]
            src_entry_indices, trg_entries = self.batch_find_trg_entries(src_ids, ref)
            decoding_strategy = DictionaryGuidedBeamSearch(
                src_entry_indices,
                trg_entries,
                decoding_strategy.beam_size,
                decoding_strategy.length_penalty,
                decoding_strategy.coverage_penalty,
                decoding_strategy.tflite_output_size,
            )

        # Dynamically decodes from the encoder outputs.
        initial_state = self.decoder.initial_state(
            memory=encoder_outputs,
            memory_sequence_length=encoder_sequence_length,
            initial_state=encoder_state,
        )
        (sampled_ids, sampled_length, log_probs, alignment, _,) = self.decoder.dynamic_decode(
            self.labels_inputter,
            start_ids,
            initial_state=initial_state,
            decoding_strategy=decoding_strategy,
            sampler=Sampler.from_params(params),
            maximum_iterations=params.get("maximum_decoding_length", 250),
            minimum_iterations=params.get("minimum_decoding_length", 0),
            tflite_output_size=params.get("tflite_output_size", 250) if tflite_run else None,
        )

        if tflite_run:
            target_tokens = sampled_ids
        else:
            target_tokens = self.labels_inputter.ids_to_tokens.lookup(tf.cast(sampled_ids, tf.int64))
        # Maybe replace unknown targets by the source tokens with the highest attention weight.
        if params.get("replace_unknown_target", False):
            if alignment is None:
                raise TypeError(
                    "replace_unknown_target is not compatible with decoders " "that don't return alignment history"
                )
            if not isinstance(self.features_inputter, WordEmbedder):
                raise TypeError("replace_unknown_target is only defined when the source " "inputter is a WordEmbedder")

            source_tokens = features if tflite_run else features["inputter_0_tokens"]
            if beam_size > 1:
                source_tokens = tfa.seq2seq.tile_batch(source_tokens, beam_size)
            original_shape = tf.shape(target_tokens)
            if tflite_run:
                target_tokens = tf.squeeze(target_tokens, axis=0)
                output_size = original_shape[-1]
                unknown_token = self.labels_inputter.vocabulary_size - 1
            else:
                target_tokens = tf.reshape(target_tokens, [-1, original_shape[-1]])
                output_size = tf.shape(target_tokens)[1]
                unknown_token = UNKNOWN_TOKEN

            align_shape = shape_list(alignment)
            attention = tf.reshape(
                alignment,
                [align_shape[0] * align_shape[1], align_shape[2], align_shape[3]],
            )
            attention = align_in_time(attention, output_size)
            replaced_target_tokens = replace_unknown_target(
                target_tokens, source_tokens, attention, unknown_token=unknown_token
            )
            if tflite_run:
                target_tokens = replaced_target_tokens
            else:
                target_tokens = tf.reshape(replaced_target_tokens, original_shape)

        if tflite_run:
            if beam_size > 1:
                target_tokens = tf.transpose(target_tokens)
                target_tokens = target_tokens[:, :1]
            target_tokens = tf.squeeze(target_tokens)

            return target_tokens
        # Maybe add noise to the predictions.
        decoding_noise = params.get("decoding_noise")
        if decoding_noise:
            target_tokens, sampled_length = _add_noise(
                target_tokens,
                sampled_length,
                decoding_noise,
                params.get("decoding_subword_token", "￭"),
                params.get("decoding_subword_token_is_spacer"),
            )
            alignment = None  # Invalidate alignments.

        predictions = {"log_probs": log_probs}
        if self.labels_inputter.tokenizer.in_graph:
            detokenized_text = self.labels_inputter.tokenizer.detokenize(
                tf.reshape(target_tokens, [batch_size * beam_size, -1]),
                sequence_length=tf.reshape(sampled_length, [batch_size * beam_size]),
            )
            predictions["text"] = tf.reshape(detokenized_text, [batch_size, beam_size])
        else:
            predictions["tokens"] = target_tokens
            predictions["length"] = sampled_length
            if alignment is not None:
                predictions["alignment"] = alignment

        # Maybe restrict the number of returned hypotheses based on the user parameter.
        num_hypotheses = params.get("num_hypotheses", 1)
        if num_hypotheses > 0:
            if num_hypotheses > beam_size:
                raise ValueError("n_best cannot be greater than beam_width")
            for key, value in predictions.items():
                predictions[key] = value[:, :num_hypotheses]
        return predictions

    def batch_find_trg_entries(self, src_ids: tf.Tensor, ref: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        src_entry_indices, trg_entries = tf.map_fn(
            lambda i, r: self.find_trg_entries(i, r),
            src_ids,
            ref,
            fn_output_signature=(
                tf.TensorSpec((None), dtype=tf.int32),
                tf.RaggedTensorSpec(shape=(None, None, None), dtype=tf.int32, row_splits_dtype=tf.int32),
            ),
        )
        return src_entry_indices, trg_entries.to_tensor()

    @tf.function
    def find_trg_entries(self, src_ids: tf.Tensor, ref: tf.Tensor) -> Tuple[tf.Tensor, tf.RaggedTensor]:
        if self._dictionary is None:
            raise ValueError("The dictionary must be initialized.")
        length = tf.shape(src_ids)[0]
        src_entry_indices = tf.TensorArray(tf.int32, size=length)
        trg_entry_lengths = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
        trg_variants = tf.TensorArray(tf.int32, size=0, dynamic_size=True, infer_shape=False)
        trg_variant_lengths = tf.TensorArray(tf.int32, size=0, dynamic_size=True)

        trg_entry_lengths = trg_entry_lengths.write(0, 0)
        i = 0
        j = 1
        k = 0
        while i < length:
            trg_entry, prefix_len, entry_refs = self._dictionary.longest_prefix(src_ids[i:])
            if prefix_len == 0:
                src_entry_indices = src_entry_indices.write(i, 0)
                i += 1
            else:
                end = i + prefix_len
                matched = tf.sets.intersection(entry_refs, ref)
                if tf.size(matched) == 0:
                    while i < end:
                        src_entry_indices = src_entry_indices.write(i, 0)
                        i += 1
                else:
                    num_variants = trg_entry.nrows()
                    trg_entry_lengths = trg_entry_lengths.write(j, num_variants)
                    while i < end:
                        src_entry_indices = src_entry_indices.write(i, j)
                        i += 1
                    j += 1
                    for vi in tf.range(num_variants):
                        trg_variant = trg_entry[vi]
                        trg_variants = trg_variants.write(k, trg_variant)
                        trg_variant_lengths = trg_variant_lengths.write(k, tf.shape(trg_variant)[0])
                        k += 1
        if k == 0:
            trg_variants = trg_variants.write(0, tf.constant([], dtype=tf.int32))
        return src_entry_indices.stack(), tf.RaggedTensor.from_nested_row_lengths(
            trg_variants.concat(), [trg_entry_lengths.stack(), trg_variant_lengths.stack()]
        )

    def transfer_weights(
        self,
        new_model: "SILTransformer",
        new_optimizer: Any = None,
        optimizer: Any = None,
        ignore_weights: Optional[List[tf.Variable]] = None,
    ):
        updated_variables = []

        def _map_variable(mapping, var_a, var_b, axis=0):
            if new_optimizer is not None and optimizer is not None:
                variables = update_variable_and_slots(
                    var_a,
                    var_b,
                    optimizer,
                    new_optimizer,
                    mapping,
                    vocab_axis=axis,
                )
            else:
                variables = [update_variable(var_a, var_b, mapping, vocab_axis=axis)]
            updated_variables.extend(variables)

        source_mapping, _ = get_mapping(
            self.features_inputter.vocabulary_file,
            new_model.features_inputter.vocabulary_file,
        )
        target_mapping, _ = get_mapping(
            self.labels_inputter.vocabulary_file,
            new_model.labels_inputter.vocabulary_file,
        )

        _map_variable(
            source_mapping,
            self.features_inputter.embedding,
            new_model.features_inputter.embedding,
        )
        _map_variable(
            target_mapping,
            self.decoder.output_layer.bias,
            new_model.decoder.output_layer.bias,
        )

        if not EmbeddingsSharingLevel.share_input_embeddings(self.share_embeddings):
            _map_variable(
                target_mapping,
                self.labels_inputter.embedding,
                new_model.labels_inputter.embedding,
            )
        if not EmbeddingsSharingLevel.share_target_embeddings(self.share_embeddings):
            _map_variable(
                target_mapping,
                self.decoder.output_layer.kernel,
                new_model.decoder.output_layer.kernel,
                axis=1,
            )

        return super(SequenceToSequence, self).transfer_weights(
            new_model,
            new_optimizer=new_optimizer,
            optimizer=optimizer,
            ignore_weights=updated_variables + (ignore_weights if ignore_weights is not None else []),
        )


@register_model_in_catalog
class SILTransformerMedium(SILTransformer):
    def __init__(self):
        super().__init__(num_layers=3)


@register_model_in_catalog(alias="SILTransformer")
class SILTransformerBase(SILTransformer):
    """Defines a Transformer model as decribed in https://arxiv.org/abs/1706.03762."""


@register_model_in_catalog
class SILTransformerBaseAlignmentEnhanced(SILTransformer):
    """Defines a Transformer model as decribed in https://arxiv.org/abs/1706.03762."""

    def __init__(self):
        super().__init__(
            source_inputter=ParallelInputter(
                [WordEmbedder(embedding_size=512), RefInputter()], reducer=TakeFirstReducer()
            ),
            attention_reduction=MultiHeadAttentionReduction.AVERAGE_ALL_LAYERS,
            alignment_head_num_units=64,
        )


@register_model_in_catalog(alias="SILTransformerRelative")
class SILTransformerBaseRelative(SILTransformer):
    """
    Defines a Transformer model using relative position representations as described in
    https://arxiv.org/abs/1803.02155.
    """

    def __init__(self):
        super().__init__(position_encoder_class=None, maximum_relative_position=20)


@register_model_in_catalog
class SILTransformerBaseNoResidual(SILTransformer):
    """
    Defines a Transformer model with no residual connection for the self-attention layer of a middle encoder layer
    as described in https://arxiv.org/abs/2012.15127.
    """

    def __init__(self):
        super().__init__(drop_encoder_self_attention_residual_connections={3})


@register_model_in_catalog
class SILTransformerBig(SILTransformer):
    """Defines a large Transformer model as decribed in https://arxiv.org/abs/1706.03762."""

    def __init__(self):
        super().__init__(num_units=1024, num_heads=16, ffn_inner_dim=4096)


@register_model_in_catalog
class SILTransformerBigRelative(SILTransformer):
    """
    Defines a large Transformer model using relative position representations as described in
    https://arxiv.org/abs/1803.02155.
    """

    def __init__(self):
        super().__init__(
            num_units=1024, num_heads=16, ffn_inner_dim=4096, position_encoder_class=None, maximum_relative_position=20
        )


@register_model_in_catalog
class SILTransformerTiny(SILTransformer):
    """Defines a tiny Transformer model."""

    def __init__(self):
        super().__init__(num_layers=2, num_units=64, num_heads=2, ffn_inner_dim=64)
