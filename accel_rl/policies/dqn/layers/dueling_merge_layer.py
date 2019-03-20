
import theano.tensor as T
from lasagne.layers.base import MergeLayer


class DuelingMergeLayer(MergeLayer):
    """
    Input the Layers into this one in the order: (Value, Advantage)

    Assumes that dim-0 is for different data entries,
    and dim-1 is the actions dimension for broadcasting and averaging.
    If needed dim-2 will correspond to distributional atoms.

    Value and Advantage must have same number of dimensions, for less
    risk of error in broadcasting.
    """

    def __init__(self, incomings, name=None):
        assert len(incomings) == 2
        super().__init__(incomings, name)
        assert len(self.input_shapes[0]) == len(self.input_shapes[1])
        assert self.input_shapes[0][0] == self.input_shapes[1][0]

    def get_output_shape_for(self, input_shapes):
        assert len(input_shapes) == 2
        shape_0, shape_1 = input_shapes
        assert len(shape_0) == len(shape_1)
        assert shape_0[1] == 1
        return shape_1

    def get_output_for(self, inputs, **kwargs):
        val, adv = inputs
        mean_adv = T.mean(adv, axis=1, keepdims=True)
        return val + (adv - mean_adv)
