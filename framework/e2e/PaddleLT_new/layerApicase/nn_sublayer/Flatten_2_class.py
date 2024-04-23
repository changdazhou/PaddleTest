import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: Flatten_2
    api简介: 展开一个tensor
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = paddle.nn.Flatten(stop_axis=-1, )

    def forward(self, data0, ):
        """
        forward
        """
        out = self.func(data0, )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-1 + (1 - -1) * np.random.random([2, 2, 3]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (1 - -1) * np.random.random([2, 2, 3]).astype('float32'), )
    return inputs

