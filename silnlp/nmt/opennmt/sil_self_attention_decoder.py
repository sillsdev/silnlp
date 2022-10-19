import tensorflow as tf
from opennmt import END_OF_SENTENCE_ID
from opennmt.decoders import SelfAttentionDecoder
from opennmt.inputters import WordEmbedder
from opennmt.layers import (
    Dense,
    FeedForwardNetwork,
    LayerNorm,
    MultiHeadAttention,
    MultiHeadAttentionReduction,
    SinusoidalPositionEncoder,
    TransformerLayerWrapper,
    dropout,
    future_mask,
    split_heads,
)

from .decoding import dynamic_decode
from .sil_transformer_layer_wrapper import SILTransformerLayerWrapper

EPSILON: float = 1e-07


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
        mha_bias=True,
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
                mha_bias=mha_bias,
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
        inputs *= self.num_units**0.5
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
        if decoding_strategy is not None and hasattr(decoding_strategy, "_set_state_reorder_flags"):
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


def clip_attention_probs(attention: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    attention = tf.clip_by_value(attention, EPSILON, 1.0 - EPSILON)
    mask = tf.cast(mask, attention.dtype)
    if mask.shape.rank == 2:
        mask = tf.expand_dims(mask, 1)  # Broadcast on time dimension.
    mask = tf.expand_dims(mask, 1)  # Broadcast on head dimension.
    attention = tf.multiply(attention, mask)
    return attention


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
        queries *= self.num_units**-0.5

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
        mha_bias=True,
        maximum_relative_position=None,
        pre_norm=True,
        alignment_head_num_units=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.self_attention = MultiHeadAttention(
            num_heads,
            num_units,
            bias=mha_bias,
            dropout=attention_dropout,
            maximum_relative_position=maximum_relative_position,
        )
        self.self_attention = TransformerLayerWrapper(self.self_attention, dropout, pre_norm=pre_norm)
        self.attention = []
        for _ in range(num_sources):
            attention = MultiHeadAttention(
                num_heads,
                num_units,
                bias=mha_bias,
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
