import tensorflow as tf
from opennmt import END_OF_SENTENCE_ID, START_OF_SENTENCE_ID
from opennmt.decoders import SelfAttentionDecoder
from opennmt.encoders import ParallelEncoder, SelfAttentionEncoder
from opennmt.inputters import WordEmbedder, add_sequence_controls
from opennmt.layers import (
    FeedForwardNetwork,
    LayerNorm,
    MultiHeadAttention,
    SelfAttentionEncoderLayer,
    SinusoidalPositionEncoder,
    TransformerLayerWrapper,
)
from opennmt.models import EmbeddingsSharingLevel, Transformer, register_model_in_catalog


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
            num_heads, num_units, dropout=attention_dropout, maximum_relative_position=maximum_relative_position,
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
                encoders if not share_encoders else encoders[0], outputs_reducer=None, states_reducer=None,
            )
        else:
            encoder = encoders[0]
        decoder = SelfAttentionDecoder(
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
        super(Transformer, self).__init__(
            source_inputter, target_inputter, encoder, decoder, share_embeddings=share_embeddings,
        )

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
