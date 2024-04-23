import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: conv3d_0
    api简介: 3维卷积
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """
        out = paddle.nn.functional.conv3d(x,  weight=paddle.to_tensor(-1 + (1 - -1) * np.random.random([3, 1, 2, 2, 2]).astype('float32'), dtype='float32', stop_gradient=False), padding=0, groups=1, )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-1 + (1 - -1) * np.random.random([3, 1, 3, 3, 3]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (1 - -1) * np.random.random([3, 1, 3, 3, 3]).astype('float32'), )
    return inputs

