import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: Conv2D_2
    api简介: 2维卷积
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = paddle.nn.Conv2D(in_channels=3, out_channels=3, kernel_size=[2, 2], stride=1, padding=0, )

    def forward(self, data, ):
        """
        forward
        """
        out = self.func(data, )
        return out



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, 3, -1, -1), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(0 + (1 - 0) * np.random.random([2, 3, 4, 4]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (0 + (1 - 0) * np.random.random([2, 3, 4, 4]).astype('float32'), )
    return inputs

