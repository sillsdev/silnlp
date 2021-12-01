import tensorflow as tf
from opennmt.encoders import SelfAttentionEncoder
from opennmt.layers import (
    FeedForwardNetwork,
    LayerNorm,
    MultiHeadAttention,
    SelfAttentionEncoderLayer,
    SinusoidalPositionEncoder,
)

from .sil_transformer_layer_wrapper import SILTransformerLayerWrapper


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
