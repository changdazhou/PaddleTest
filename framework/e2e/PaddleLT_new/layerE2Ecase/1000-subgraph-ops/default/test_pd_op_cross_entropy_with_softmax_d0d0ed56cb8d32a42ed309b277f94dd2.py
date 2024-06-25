import os
if os.getenv('FLAGS_cinn_new_group_scheduler') is None:
    os.environ['FLAGS_cinn_new_group_scheduler'] = '1'
if os.getenv('FLAGS_group_schedule_tiling_first') is None:
    os.environ['FLAGS_group_schedule_tiling_first'] = '1'
if os.getenv('FLAGS_prim_all') is None:
    os.environ['FLAGS_prim_all'] = 'true'
if os.getenv('FLAGS_prim_enable_dynamic') is None:
    os.environ['FLAGS_prim_enable_dynamic'] = '1'
if os.getenv('FLAGS_enable_pir_api') is None:
    os.environ['FLAGS_enable_pir_api'] = '1'
if os.getenv('FLAGS_cinn_bucket_compile') is None:
    os.environ['FLAGS_cinn_bucket_compile'] = '1'

import unittest
import numpy as np
import paddle

def GetEnvVarEnableJit():
    enable_jit = os.getenv('PADDLE_DEBUG_ENABLE_JIT')
    return enable_jit not in {
        "0",
        "False",
        "false",
        "OFF",
    }

def GetEnvVarEnableCinn():
    enable_cinn = os.getenv('PADDLE_DEBUG_ENABLE_CINN')
    if enable_cinn is None:
        return True
    return enable_cinn not in {
        "0",
        "False",
        "false",
        "OFF",
    }


def GetTolerance(dtype):
    if dtype == np.float16:
        return GetFloat16Tolerance()
    if dtype == np.float32:
        return GetFloat32Tolerance()
    return 1e-6

def GetFloat16Tolerance():
    try:
        return float(os.getenv('PADDLE_DEBUG_FLOAT16_TOL'))
    except:
        return 1e-3

def GetFloat32Tolerance():
    try:
        return float(os.getenv('PADDLE_DEBUG_FLOAT32_TOL'))
    except:
        return 1e-6

def IsInteger(dtype):
    return np.dtype(dtype).char in np.typecodes['AllInteger']

def ApplyToStatic(net, use_cinn):
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = use_cinn
    return paddle.jit.to_static(
        net,
        input_spec=net.get_input_spec(),
        build_strategy=build_strategy,
        full_graph=True,
    )

class InstanceTrait:

    @classmethod
    def instance(cls):
        if cls.instance_ is None:
            cls.instance_ = cls()
        return cls.instance_

    @classmethod
    def static_instance_with_cinn(cls):
        if cls.static_instance_with_cinn_ is None:
            cls.static_instance_with_cinn_ = ApplyToStatic(
                cls.instance(),
                use_cinn=True
            )
        return cls.static_instance_with_cinn_

    @classmethod
    def static_instance_without_cinn(cls):
        if cls.static_instance_without_cinn_ is None:
            cls.static_instance_without_cinn_ = ApplyToStatic(
                cls.instance(),
                use_cinn=False
            )
        return cls.static_instance_without_cinn_


class CinnTestBase:

    def setUp(self):
        paddle.seed(2024)
        self.prepare_data()

    def test_train(self):
        dy_outs = self.train(use_cinn=False)
        cinn_outs = self.train(use_cinn=GetEnvVarEnableCinn())

        for cinn_out, dy_out in zip(cinn_outs, dy_outs):
          if type(cinn_out) is list and type(dy_out) is list:
            for x, y in zip(cinn_out, dy_out):
              self.assert_all_close(x, y)
          else:
            self.assert_all_close(cinn_out, dy_out)

    def train(self, use_cinn):
        if GetEnvVarEnableJit():
            net = self.prepare_static_net(use_cinn)
        else:
            net = self.prepare_net()
        out = net(*self.inputs)
        return out
    
    def prepare_data(self):
        self.inputs = self.get_inputs()
        for input in self.inputs:
            input.stop_gradient = True

    def prepare_net(self):
        return self.get_test_class().instance()

    def prepare_static_net(self, use_cinn):
        if use_cinn:
            return self.get_test_class().static_instance_with_cinn()
        else:
            return self.get_test_class().static_instance_without_cinn()

    def assert_all_close(self, x, y):
        if (hasattr(x, "numpy") and hasattr(y, "numpy")):
            x_numpy = x.numpy()
            y_numpy = y.numpy()
            assert x_numpy.dtype == y_numpy.dtype
            if IsInteger(x_numpy.dtype):
                np.testing.assert_equal(x_numpy, y_numpy)
            else:
                tol = GetTolerance(x_numpy.dtype)
                np.testing.assert_allclose(x_numpy, y_numpy, atol=tol, rtol=tol)
        else:
            assert x == y



class PrimitiveOp_038c269422135632bbc6046c3478ad49(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0c14eee62dfb0818875dbc68a483dc99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_038c269422135632bbc6046c3478ad49
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]], dtype='int64').reshape([16, 1]),
        ]


class TestPrimitiveOp_4c3a052b833dadc50c3eee7652fd3952(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_038c269422135632bbc6046c3478ad49
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]], dtype='int64').reshape([16, 1]),
        ]


class PrimitiveOp_134762e5a722f81a7936be279bc39270(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 4, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e16c59e75ed7491f16e0fe5a1f19eb11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_134762e5a722f81a7936be279bc39270
    def get_inputs(self):
        return [
            paddle.uniform([1799, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[1799, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_e16c59e75ed7491f16e0fe5a1f19eb11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_134762e5a722f81a7936be279bc39270
    def get_inputs(self):
        return [
            paddle.uniform([1799, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[1799, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_89c3f86de131085b10cfd56b6f578a17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_134762e5a722f81a7936be279bc39270
    def get_inputs(self):
        return [
            paddle.uniform([5504, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[5504, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_89c3f86de131085b10cfd56b6f578a17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_134762e5a722f81a7936be279bc39270
    def get_inputs(self):
        return [
            paddle.uniform([5504, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[5504, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_2188741a559bc2f0221da073f10e9d8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_038c269422135632bbc6046c3478ad49
    def get_inputs(self):
        return [
            paddle.uniform([36, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[36, 1], dtype='int64'),
        ]


class TestPrimitiveOp_2188741a559bc2f0221da073f10e9d8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_038c269422135632bbc6046c3478ad49
    def get_inputs(self):
        return [
            paddle.uniform([36, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[36, 1], dtype='int64'),
        ]


class PrimitiveOp_00f204ea65841ce67bf69ea3ce82366f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, 19], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 4, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3972e011d1ce02e88fc7c82a1accdbb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00f204ea65841ce67bf69ea3ce82366f
    def get_inputs(self):
        return [
            paddle.uniform([1811, 4, 19], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[1811, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_3972e011d1ce02e88fc7c82a1accdbb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00f204ea65841ce67bf69ea3ce82366f
    def get_inputs(self):
        return [
            paddle.uniform([1811, 4, 19], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[1811, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_967e6b725ae0316114d3eeb24e25b524(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_038c269422135632bbc6046c3478ad49
    def get_inputs(self):
        return [
            paddle.uniform([24, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]], dtype='int64').reshape([24, 1]),
        ]


class TestPrimitiveOp_69525c8017185a4b8ac8da416ec283b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_038c269422135632bbc6046c3478ad49
    def get_inputs(self):
        return [
            paddle.uniform([24, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]], dtype='int64').reshape([24, 1]),
        ]


class TestPrimitiveOp_501d5b7402bbf001e0c223a6e3e4cd7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_134762e5a722f81a7936be279bc39270
    def get_inputs(self):
        return [
            paddle.uniform([1559, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[1559, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_501d5b7402bbf001e0c223a6e3e4cd7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_134762e5a722f81a7936be279bc39270
    def get_inputs(self):
        return [
            paddle.uniform([1559, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[1559, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_147aafcc87b57d4c0a6acb9f7394fd29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_038c269422135632bbc6046c3478ad49
    def get_inputs(self):
        return [
            paddle.uniform([4, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0], [0], [0], [0]], dtype='int64').reshape([4, 1]),
        ]


class TestPrimitiveOp_8d7a664ddc9948c760c67f93115520f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_038c269422135632bbc6046c3478ad49
    def get_inputs(self):
        return [
            paddle.uniform([4, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[1], [1], [1], [1]], dtype='int64').reshape([4, 1]),
        ]


class TestPrimitiveOp_ad99028e0f36c4117dd9d83a1fc74c1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_134762e5a722f81a7936be279bc39270
    def get_inputs(self):
        return [
            paddle.uniform([2066, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[2066, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_ad99028e0f36c4117dd9d83a1fc74c1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_134762e5a722f81a7936be279bc39270
    def get_inputs(self):
        return [
            paddle.uniform([2066, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[2066, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_aeb9c06e72e79050d687c29ad175bb6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_134762e5a722f81a7936be279bc39270
    def get_inputs(self):
        return [
            paddle.uniform([4618, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[4618, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_aeb9c06e72e79050d687c29ad175bb6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_134762e5a722f81a7936be279bc39270
    def get_inputs(self):
        return [
            paddle.uniform([4618, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[4618, 4, 1], dtype='int64'),
        ]


class PrimitiveOp_223747de6c518b4d572ea6e7a776912a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_60559936d345700cd597790c07136ce8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_223747de6c518b4d572ea6e7a776912a
    def get_inputs(self):
        return [
            paddle.uniform([1, 2434, 81], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[1, 2434, 1], dtype='int64'),
        ]


class TestPrimitiveOp_cae4b4fd5b0c854358d91b7e7cb8b04b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_134762e5a722f81a7936be279bc39270
    def get_inputs(self):
        return [
            paddle.uniform([1058, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[1058, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_cae4b4fd5b0c854358d91b7e7cb8b04b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_134762e5a722f81a7936be279bc39270
    def get_inputs(self):
        return [
            paddle.uniform([1058, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[1058, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_53bd165fe47d635774c241d0a32a08d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_134762e5a722f81a7936be279bc39270
    def get_inputs(self):
        return [
            paddle.uniform([2402, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[2402, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_53bd165fe47d635774c241d0a32a08d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_134762e5a722f81a7936be279bc39270
    def get_inputs(self):
        return [
            paddle.uniform([2402, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[2402, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_eb00708fc71dc02d8c0aeca6c41582d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_134762e5a722f81a7936be279bc39270
    def get_inputs(self):
        return [
            paddle.uniform([2993, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[2993, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_eb00708fc71dc02d8c0aeca6c41582d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_134762e5a722f81a7936be279bc39270
    def get_inputs(self):
        return [
            paddle.uniform([2993, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[2993, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_f89ef2c1c6a8535b4f0c79bea3a8d03a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_134762e5a722f81a7936be279bc39270
    def get_inputs(self):
        return [
            paddle.uniform([3787, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[3787, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_f89ef2c1c6a8535b4f0c79bea3a8d03a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_134762e5a722f81a7936be279bc39270
    def get_inputs(self):
        return [
            paddle.uniform([3787, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[3787, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_be27abbbf446a68c9a1d677c6165e1d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_038c269422135632bbc6046c3478ad49
    def get_inputs(self):
        return [
            paddle.uniform([20, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]], dtype='int64').reshape([20, 1]),
        ]


class TestPrimitiveOp_a183b67e033ccd8e840000e5e873e191(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_038c269422135632bbc6046c3478ad49
    def get_inputs(self):
        return [
            paddle.uniform([20, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]], dtype='int64').reshape([20, 1]),
        ]


class TestPrimitiveOp_b80743b89620be2e30d01b4545d99563(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_223747de6c518b4d572ea6e7a776912a
    def get_inputs(self):
        return [
            paddle.uniform([1, 8732, 21], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[1, 8732, 1], dtype='int64'),
        ]


class TestPrimitiveOp_529041a6fa04052e0affb0fb38a9bd81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_134762e5a722f81a7936be279bc39270
    def get_inputs(self):
        return [
            paddle.uniform([2114, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[2114, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_529041a6fa04052e0affb0fb38a9bd81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_134762e5a722f81a7936be279bc39270
    def get_inputs(self):
        return [
            paddle.uniform([2114, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[2114, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_38b65057366cd826ddff93325ace3c5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_134762e5a722f81a7936be279bc39270
    def get_inputs(self):
        return [
            paddle.uniform([4156, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[4156, 4, 1], dtype='int64'),
        ]


class TestPrimitiveOp_38b65057366cd826ddff93325ace3c5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_134762e5a722f81a7936be279bc39270
    def get_inputs(self):
        return [
            paddle.uniform([4156, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[4156, 4, 1], dtype='int64'),
        ]




if __name__ == '__main__':
    unittest.main()