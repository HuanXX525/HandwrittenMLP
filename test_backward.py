import unittest
import math
import numpy as np
from backward import Value, ActivationFunction, LossFunction, Optimizer, Neuron, Layer


class TestValue(unittest.TestCase):
    """测试Value类的基本运算和反向传播"""

    def test_addition(self):
        """测试加法运算"""
        a = Value(2.0)
        b = Value(3.0)
        c = a + b
        self.assertEqual(c.data, 5.0)
        
        # 测试反向传播
        c.backward()
        self.assertEqual(a.grad, 1.0)
        self.assertEqual(b.grad, 1.0)

    def test_multiplication(self):
        """测试乘法运算"""
        a = Value(2.0)
        b = Value(3.0)
        c = a * b
        self.assertEqual(c.data, 6.0)
        
        # 测试反向传播
        c.backward()
        self.assertEqual(a.grad, 3.0)
        self.assertEqual(b.grad, 2.0)

    def test_pow(self):
        """测试幂运算"""
        a = Value(2.0)
        b = Value(3.0)
        c = a ** b
        self.assertAlmostEqual(c.data, 8.0)
        
        # 测试反向传播
        c.backward()
        self.assertAlmostEqual(a.grad, 3.0 * 2.0**2.0)
        self.assertAlmostEqual(b.grad, 8.0 * math.log(2.0))

    def test_division(self):
        """测试除法运算"""
        a = Value(6.0)
        b = Value(3.0)
        c = a / b
        self.assertEqual(c.data, 2.0)
        
        # 测试反向传播
        c.backward()
        self.assertAlmostEqual(a.grad, 1.0/3.0)
        self.assertAlmostEqual(b.grad, -6.0/9.0)

    def test_subtraction(self):
        """测试减法运算"""
        a = Value(5.0)
        b = Value(3.0)
        c = a - b
        self.assertEqual(c.data, 2.0)
        
        # 测试反向传播
        c.backward()
        self.assertEqual(a.grad, 1.0)
        self.assertEqual(b.grad, -1.0)

    def test_negation(self):
        """测试负号运算"""
        a = Value(5.0)
        b = -a
        self.assertEqual(b.data, -5.0)
        
        # 测试反向传播
        b.backward()
        self.assertEqual(a.grad, -1.0)

    def test_tanh(self):
        """测试tanh运算"""
        a = Value(1.0)
        b = a.tanh()
        expected = math.tanh(1.0)
        self.assertAlmostEqual(b.data, expected)
        
        # 测试反向传播
        b.backward()
        self.assertAlmostEqual(a.grad, 1.0 - expected**2)

    def test_exp(self):
        """测试exp运算"""
        a = Value(1.0)
        b = a.exp()
        expected = math.exp(1.0)
        self.assertAlmostEqual(b.data, expected)
        
        # 测试反向传播
        b.backward()
        self.assertAlmostEqual(a.grad, expected)

    def test_log(self):
        """测试log运算"""
        a = Value(2.0)
        b = a.log()
        expected = math.log(2.0)
        self.assertAlmostEqual(b.data, expected)
        
        # 测试反向传播
        b.backward()
        self.assertAlmostEqual(a.grad, 1.0/2.0)

    def test_max(self):
        """测试max运算"""
        a = Value(2.0)
        b = Value(3.0)
        c = a.max(b)
        self.assertEqual(c.data, 3.0)
        
        # 测试反向传播
        c.backward()
        self.assertEqual(a.grad, 0.0)
        self.assertEqual(b.grad, 1.0)

    def test_complex_expression(self):
        """测试复杂表达式"""
        a = Value(2.0)
        b = Value(3.0)
        c = a * b + a ** 2
        self.assertEqual(c.data, 10.0)
        
        # 测试反向传播
        c.backward()
        self.assertEqual(a.grad, 7.0)  # b + 2*a = 3 + 4 = 7
        self.assertEqual(b.grad, 2.0)  # a = 2


class TestActivationFunction(unittest.TestCase):
    """测试激活函数"""

    def test_relu(self):
        """测试ReLU激活函数"""
        inputs = np.array([Value(-1.0), Value(0.0), Value(1.0), Value(2.0)])
        outputs = ActivationFunction.RelU(inputs)
        
        self.assertEqual(outputs[0].data, 0.0)
        self.assertEqual(outputs[1].data, 0.0)
        self.assertEqual(outputs[2].data, 1.0)
        self.assertEqual(outputs[3].data, 2.0)

    def test_softmax(self):
        """测试Softmax激活函数"""
        inputs = np.array([Value(1.0), Value(2.0), Value(3.0)])
        outputs = ActivationFunction.softmax(inputs)
        
        # 计算期望值
        exp_values = [math.exp(1.0), math.exp(2.0), math.exp(3.0)]
        sum_exp = sum(exp_values)
        expected = [ev/sum_exp for ev in exp_values]
        
        for i in range(len(outputs)):
            self.assertAlmostEqual(outputs[i].data, expected[i])


class TestLossFunction(unittest.TestCase):
    """测试损失函数"""

    def test_binary_cross_entropy(self):
        """测试二元交叉熵损失"""
        y_hat = Value(0.8)
        y_true = 1.0
        loss = LossFunction.binary_cross_entropy(y_hat, y_true)
        
        # 计算期望值: -log(0.8)
        expected = -math.log(0.8)
        self.assertAlmostEqual(loss.data, expected)

    def test_categorical_cross_entropy(self):
        """测试类别交叉熵损失"""
        y_hat = np.array([Value(0.1), Value(0.8), Value(0.1)])
        y_true = np.array([0.0, 1.0, 0.0])
        loss = LossFunction.categorical_cross_entropy(y_hat, y_true)
        
        # 计算期望值: -log(0.8)
        expected = -math.log(0.8)
        self.assertAlmostEqual(loss.data, expected)


class TestOptimizer(unittest.TestCase):
    """测试优化器"""

    def test_sgd(self):
        """测试SGD优化器"""
        a = Value(2.0)
        b = Value(3.0)
        params = [a, b]
        
        # 执行一次前向传播和反向传播
        c = a * b
        c.backward()
        
        # 使用SGD优化器更新参数
        optimizer = Optimizer(params, alpha=0.1)
        optimizer.step()
        
        # 检查参数是否正确更新
        self.assertAlmostEqual(a.data, 2.0 - 0.1 * 3.0)
        self.assertAlmostEqual(b.data, 3.0 - 0.1 * 2.0)

    def test_zero_grad(self):
        """测试梯度清零"""
        a = Value(2.0)
        b = Value(3.0)
        params = [a, b]
        
        # 执行一次前向传播和反向传播
        c = a * b
        c.backward()
        
        # 检查梯度是否正确计算
        self.assertEqual(a.grad, 3.0)
        self.assertEqual(b.grad, 2.0)
        
        # 清零梯度
        optimizer = Optimizer(params)
        optimizer.zero_grad()
        
        # 检查梯度是否清零
        self.assertEqual(a.grad, 0.0)
        self.assertEqual(b.grad, 0.0)


class TestNeuronAndLayer(unittest.TestCase):
    """测试神经元和层"""

    def test_neuron(self):
        """测试神经元"""
        # 创建一个输入维度为3的神经元
        neuron = Neuron(3)
        
        # 创建输入数据
        x = np.array([Value(1.0), Value(2.0), Value(3.0)])
        
        # 前向传播
        output = neuron(x)
        
        # 检查输出类型
        self.assertIsInstance(output, Value)
        
        # 检查参数数量
        params = neuron.parameters()
        self.assertEqual(len(params), 4)  # 3个权重 + 1个偏置

    def test_layer(self):
        """测试层"""
        # 创建一个输入维度为3，输出维度为2的层
        layer = Layer(3, 2)
        
        # 创建输入数据
        x = np.array([Value(1.0), Value(2.0), Value(3.0)])
        
        # 前向传播
        outputs = layer(x)
        
        # 检查输出形状
        self.assertEqual(len(outputs), 2)
        
        # 检查输出类型
        for output in outputs:
            self.assertIsInstance(output, Value)
        
        # 检查参数数量
        params = layer.parameters()
        # 每个神经元有3个权重和1个偏置，共2个神经元
        self.assertEqual(len(params), 2 * (3 + 1))


if __name__ == "__main__":
    unittest.main()