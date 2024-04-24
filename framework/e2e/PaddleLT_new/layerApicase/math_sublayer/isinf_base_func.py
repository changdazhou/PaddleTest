import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: isinf_base
    api简介: 返回输入tensor的每一个值是否为 +/-INF
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """
        out = paddle.isinf(x,  )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = ()
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (paddle.to_tensor([[-1.0, 2.0, 'nan'], ['-inf', 'inf', '-nan'], [2.4, 0.0, '-inf']], dtype='float32', stop_gradient=False), )
    return inputs

