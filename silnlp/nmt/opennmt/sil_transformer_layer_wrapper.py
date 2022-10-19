from opennmt.layers import TransformerLayerWrapper


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
