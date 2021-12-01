from opennmt.layers import Reducer


class TakeFirstReducer(Reducer):
    def reduce(self, inputs):
        return inputs[0]

    def reduce_sequence(self, inputs, sequence_lengths):
        return inputs[0], sequence_lengths[0]
