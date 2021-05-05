from typing import List, Optional

import tensorflow as tf
import tensorflow_addons as tfa
from opennmt import END_OF_SENTENCE_ID, START_OF_SENTENCE_ID
from opennmt.decoders import SelfAttentionDecoder
from opennmt.encoders import ParallelEncoder, SelfAttentionEncoder
from opennmt.inputters import WordEmbedder, add_sequence_controls
from opennmt.layers import (
    Dense,
    FeedForwardNetwork,
    LayerNorm,
    LayerWrapper,
    MultiHeadAttention,
    SelfAttentionEncoderLayer,
    SinusoidalPositionEncoder,
    TransformerLayerWrapper,
    dropout,
    future_mask,
)
from opennmt.layers.reducer import align_in_time
from opennmt.models import EmbeddingsSharingLevel, Transformer, register_model_in_catalog
from opennmt.models.sequence_to_sequence import _add_noise, replace_unknown_target
from opennmt.utils.decoding import BeamSearch, DecodingStrategy, Sampler
from opennmt.utils.misc import shape_list
from pygtrie import Trie

from .decoding import DictionaryGuidedBeamSearch


@tf.function
def find_dict_trg_words(dictionary: Trie, src_ids: tf.Tensor) -> tf.RaggedTensor:
    src_ids_np = src_ids.numpy()
    i = 0
    trg_words: List[tf.Tensor] = []
    while i < len(src_ids_np):
        result = dictionary.longest_prefix(src_ids_np[i:])
        if result.is_set:
            trg_words.append(result.value)
            i += len(result.key)
        else:
            trg_words.append(tf.constant([], dtype=tf.int64))
            i += 1
    output: tf.RaggedTensor = tf.ragged.stack(trg_words)
    return output.with_row_splits_dtype(tf.int64)


class SILTransformerLayerWrapper(TransformerLayerWrapper):
    def __init__(self, layer, output_dropout, residual_connection=True, **kwargs):
        super(TransformerLayerWrapper, self).__init__(
            layer,
            normalize_input=True,
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
            self.self_attention, dropout, residual_connection=self_attention_residual_connection
        )
        self.ffn = FeedForwardNetwork(ffn_inner_dim, num_units, dropout=ffn_dropout, activation=ffn_activation)
        self.ffn = SILTransformerLayerWrapper(self.ffn, dropout)


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
        drop_self_attention_residual_connections=set(),
        **kwargs,
    ):
        super(SelfAttentionEncoder, self).__init__(**kwargs)
        self.num_units = num_units
        self.dropout = dropout
        self.position_encoder = None
        if position_encoder_class is not None:
            self.position_encoder = position_encoder_class()
        self.layer_norm = LayerNorm()
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
                self_attention_residual_connection=i not in drop_self_attention_residual_connections,
            )
            for i in range(num_layers)
        ]


class AlignmentAttention(tf.keras.layers.Layer):
    def __init__(self, num_units, **kwargs):
        super().__init__(**kwargs)
        self.num_units = num_units
        self.linear_queries = Dense(self.num_units)
        self.linear_keys = Dense(self.num_units)

    def call(self, inputs, memory=None, mask=None, cache=None):
        # Compute queries.
        queries = self.linear_queries(inputs)
        queries *= self.num_units ** -0.5

        # Compute keys.
        if cache:
            keys = tf.cond(
                tf.equal(tf.shape(cache)[1], 0),
                true_fn=lambda: self.linear_keys(memory),
                false_fn=lambda: cache,
            )
        else:
            keys = self.linear_keys(memory)

        cache = keys

        # Dot product attention.
        dot = tf.matmul(queries, keys, transpose_b=True)
        if mask is not None:
            mask = tf.cast(mask, tf.float32)
            if mask.shape.rank == 2:
                mask = tf.expand_dims(mask, 1)  # Broadcast on time dimension.
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
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.self_attention = MultiHeadAttention(
            num_heads,
            num_units,
            dropout=attention_dropout,
            maximum_relative_position=maximum_relative_position,
        )
        self.self_attention = TransformerLayerWrapper(self.self_attention, dropout)
        self.attention = []
        for i in range(num_sources):
            attention = MultiHeadAttention(num_heads, num_units, dropout=attention_dropout)
            attention = TransformerLayerWrapper(attention, dropout)
            self.attention.append(attention)
        self.alignment_attention = AlignmentAttention(num_units // num_heads)
        self.alignment_attention = LayerWrapper(self.alignment_attention, normalize_input=True)
        self.ffn = FeedForwardNetwork(ffn_inner_dim, num_units, dropout=ffn_dropout, activation=ffn_activation)
        self.ffn = TransformerLayerWrapper(self.ffn, dropout)

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

        attention = None
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
                outputs, memory_kv_i = result
                memory_kv.append(memory_kv_i)

            attention, alignment_memory_k = self.alignment_attention(
                outputs, memory=memory[0], mask=memory_mask[0], cache=cache.get("alignment_memory_k")
            )

        outputs = self.ffn(outputs, training=training)
        cache = dict(self_kv=self_kv, memory_kv=memory_kv, alignment_memory_k=alignment_memory_k)
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
        **kwargs,
    ):
        super(SelfAttentionDecoder, self).__init__(num_sources=num_sources, **kwargs)
        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout = dropout
        self.position_encoder = None
        if position_encoder_class is not None:
            self.position_encoder = position_encoder_class()
        self.layer_norm = LayerNorm()
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
            )
            for i in range(num_layers)
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

        # Run each layer.
        new_cache = []
        layer_attention = []
        for i, layer in enumerate(self.layers):
            inputs, layer_cache, attn = layer(
                inputs,
                mask=mask,
                memory=memory,
                memory_mask=memory_mask,
                cache=cache[i] if cache is not None else None,
                training=training,
            )
            new_cache.append(layer_cache)
            layer_attention.append(attn)
        outputs = self.layer_norm(inputs)
        attention = tf.stack(layer_attention, axis=-1)
        attention = tf.math.reduce_mean(attention, axis=-1)
        return outputs, new_cache, attention


class SILTransformer(Transformer):
    def __init__(
        self,
        source_inputter,
        target_inputter,
        num_layers,
        num_units,
        num_heads,
        ffn_inner_dim,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        ffn_activation=tf.nn.relu,
        position_encoder_class=SinusoidalPositionEncoder,
        share_embeddings=EmbeddingsSharingLevel.NONE,
        share_encoders=False,
        maximum_relative_position=None,
        drop_encoder_self_attention_residual_connections=set(),
    ):
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
        )

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
        src_dict_path: Optional[str] = data_config.get("src_dictionary")
        trg_dict_path: Optional[str] = data_config.get("trg_dictionary")
        if src_dict_path is not None and trg_dict_path is not None:
            self.labels_inputter.set_decoder_mode(enable=False, mark_start=False, mark_end=False)
            dictionary = Trie()
            with tf.io.gfile.GFile(src_dict_path) as src_dict, tf.io.gfile.GFile(trg_dict_path) as trg_dict:
                for src_entry, trg_entry in zip(src_dict, trg_dict):
                    src_tokens = self.features_inputter.make_features(tf.constant(src_entry.strip()))
                    trg_tokens = self.labels_inputter.make_features(tf.constant(trg_entry.strip()))
                    dictionary[src_tokens["ids"].numpy()] = trg_tokens["ids"]
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

    def _dynamic_decode(self, features, encoder_outputs, encoder_state, encoder_sequence_length):
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

        decoding_strategy = DecodingStrategy.from_params(params)
        if self._dictionary is not None and isinstance(decoding_strategy, BeamSearch):
            ids: tf.Tensor = features["ids"]
            dict_trg_words: tf.RaggedTensor = tf.map_fn(
                lambda s: find_dict_trg_words(self._dictionary, s),
                ids,
                fn_output_signature=tf.RaggedTensorSpec(shape=(None, None), dtype=tf.int64),
            )
            decoding_strategy = DictionaryGuidedBeamSearch(
                dict_trg_words,
                decoding_strategy.beam_size,
                decoding_strategy.length_penalty,
                decoding_strategy.coverage_penalty,
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
        )
        target_tokens = self.labels_inputter.ids_to_tokens.lookup(tf.cast(sampled_ids, tf.int64))

        # Maybe replace unknown targets by the source tokens with the highest attention weight.
        if params.get("replace_unknown_target", False):
            if alignment is None:
                raise TypeError(
                    "replace_unknown_target is not compatible with decoders " "that don't return alignment history"
                )
            if not isinstance(self.features_inputter, WordEmbedder):
                raise TypeError("replace_unknown_target is only defined when the source " "inputter is a WordEmbedder")
            source_tokens = features["tokens"]
            if beam_size > 1:
                source_tokens = tfa.seq2seq.tile_batch(source_tokens, beam_size)
            # Merge batch and beam dimensions.
            original_shape = tf.shape(target_tokens)
            target_tokens = tf.reshape(target_tokens, [-1, original_shape[-1]])
            align_shape = shape_list(alignment)
            attention = tf.reshape(
                alignment,
                [align_shape[0] * align_shape[1], align_shape[2], align_shape[3]],
            )
            # We don't have attention for </s> but ensure that the attention time dimension matches
            # the tokens time dimension.
            attention = align_in_time(attention, tf.shape(target_tokens)[1])
            replaced_target_tokens = replace_unknown_target(target_tokens, source_tokens, attention)
            target_tokens = tf.reshape(replaced_target_tokens, original_shape)

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


@register_model_in_catalog
class SILTransformerMedium(SILTransformer):
    def __init__(self):
        super().__init__(
            source_inputter=WordEmbedder(embedding_size=512),
            target_inputter=WordEmbedder(embedding_size=512),
            num_layers=3,
            num_units=512,
            num_heads=8,
            ffn_inner_dim=2048,
        )


@register_model_in_catalog(alias="SILTransformer")
class SILTransformerBase(SILTransformer):
    """Defines a Transformer model as decribed in https://arxiv.org/abs/1706.03762."""

    def __init__(self):
        super().__init__(
            source_inputter=WordEmbedder(embedding_size=512),
            target_inputter=WordEmbedder(embedding_size=512),
            num_layers=6,
            num_units=512,
            num_heads=8,
            ffn_inner_dim=2048,
        )


@register_model_in_catalog(alias="SILTransformerRelative")
class SILTransformerBaseRelative(SILTransformer):
    """
    Defines a Transformer model using relative position representations as described in
    https://arxiv.org/abs/1803.02155.
    """

    def __init__(self):
        super().__init__(
            source_inputter=WordEmbedder(embedding_size=512),
            target_inputter=WordEmbedder(embedding_size=512),
            num_layers=6,
            num_units=512,
            num_heads=8,
            ffn_inner_dim=2048,
            position_encoder_class=None,
            maximum_relative_position=20,
        )


@register_model_in_catalog
class SILTransformerBaseNoResidual(SILTransformer):
    """
    Defines a Transformer model with no residual connection for the self-attention layer of a middle encoder layer
    as described in https://arxiv.org/abs/2012.15127.
    """

    def __init__(self):
        super().__init__(
            source_inputter=WordEmbedder(embedding_size=512),
            target_inputter=WordEmbedder(embedding_size=512),
            num_layers=6,
            num_units=512,
            num_heads=8,
            ffn_inner_dim=2048,
            drop_encoder_self_attention_residual_connections={3},
        )


@register_model_in_catalog
class SILTransformerBig(SILTransformer):
    """Defines a large Transformer model as decribed in https://arxiv.org/abs/1706.03762."""

    def __init__(self):
        super().__init__(
            source_inputter=WordEmbedder(embedding_size=1024),
            target_inputter=WordEmbedder(embedding_size=1024),
            num_layers=6,
            num_units=1024,
            num_heads=16,
            ffn_inner_dim=4096,
        )


@register_model_in_catalog
class SILTransformerBigRelative(SILTransformer):
    """
    Defines a large Transformer model using relative position representations as described in
    https://arxiv.org/abs/1803.02155.
    """

    def __init__(self):
        super().__init__(
            source_inputter=WordEmbedder(embedding_size=1024),
            target_inputter=WordEmbedder(embedding_size=1024),
            num_layers=6,
            num_units=1024,
            num_heads=16,
            ffn_inner_dim=4096,
            position_encoder_class=None,
            maximum_relative_position=20,
        )
