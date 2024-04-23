import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: UpsamplingNearest2D_1
    api简介: 该OP用于最近邻插值插值调整一个batch中图片的大小
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = paddle.nn.UpsamplingNearest2D(size=[12, 12], data_format='NHWC', )

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
    inputs = (paddle.to_tensor(-10 + (10 - -10) * np.random.random([2, 3, 6, 10]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-10 + (10 - -10) * np.random.random([2, 3, 6, 10]).astype('float32'), )
    return inputs

