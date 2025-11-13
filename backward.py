from abc import ABC, abstractmethod
from collections import deque
import math
from typing import Callable, List, Optional
import numpy as np


def forNotValue(fun):
    '''
        如果传入的不是Value类型，则自动转为Value
    '''
    def wrapper(*args):
        args = [arg if isinstance(arg, Value) else Value(arg) for arg in args]
        return fun(*args)
    return wrapper
class Value():
    def __init__(self, data, child=())->None:
        '''
        data:当前变量的数值
        child:运算数（用于链式法则梯度计算）
        不同运算的区分可以在不同的函数里设置不同的_backward实现
        '''
        self.data = data
        self._child = child
        self.grad = 0.0
        self._backward: Optional[Callable[[], None]] = None # 会计算_child的梯度

    def __repr__(self) -> str:
        return f"Value({self.data})"



# ======== 加法运算 =======
    @forNotValue
    def __add__(self, other)-> "Value":

        out = Value(self.data + other.data, (self, other))
        
        def _backward():
            '''
                out = self + other
                Dout_self = 1
                Dout_other = 1
                一元求导法则：
                Dself = Dout_self * Dout = Dout
                Dother = Dout_other * Dout = Dout
                多元是将多个一元求导法则的Dout加起来:
                Dself += Dout
                Dother += Dout
            '''
            self.grad += out.grad
            other.grad += out.grad
        
        out._backward = _backward # (O)
        return out
    def __radd__(self, other)-> "Value":
        '''
            反向加法为了应对以下情形
            a:Vlaue, b:float
            a + b = 会正常调用__add__
            但是b+a会首先调用float的__add__，但是float的__add__并不包含Value的情况
            这时解释器会尝试调用a的__radd__
        '''
        return self + other
    
    # def __iadd__(self, other):
    #     return self + other
    
# ======== 减法运算 =======
    def __sub__(self, other)-> "Value":
        return self + (-other)

    def __rsub__(self, other)-> "Value":
        return -self + other

# ======== 负运算 =======
    def __neg__(self)-> "Value":
        return self * -1


# ======== 乘法运算 =======
    @forNotValue
    def __mul__(self, other)-> "Value":
        out = Value(self.data * other.data, (self, other))
        
        def _backward():
            '''
                out = self * other
                Dout_self = other
                Dout_other = self
                一元求导法则：
                Dself = Dout_self * Dout = other*Dout
                Dother = Dout_other * Dout = self*Dout
                多元相加
            '''
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        
        out._backward = _backward # (O)
        return out
    
    def __rmul__(self, other)-> "Value":
        return self * other
    
# ======== 指数运算和幂运算 =======
    @forNotValue
    def __pow__(self, other)-> "Value":
        assert isinstance(other.data, (float, int)), "指数运算的指数必须是int或float类型"
        y = self.data**other.data
        out = Value(y, (self, other))

        def _backward():
            '''
                out = self ** other
                Dout_self = other * self**(other - 1)
                Dout_other = self**other * ln(self)
                一元求导法则：
                Dself = Dout_self * Dout = other * self**(other - 1) * Dout
                Dother = Dout_other * Dout = self**other * ln(self) * Dout
            '''
            self.grad += other.data * self.data**(other.data - 1) * out.grad
            other.grad += y * math.log(self.data, math.e) * out.grad
        
        out._backward = _backward # (O)
        return out

# ======== 除法运算 =======
    def __truediv__(self, other)-> "Value":
        return self * other**-1

    def __rtruediv__(self, other)-> "Value":
        return other * self**-1

# ======= tanh =======
    def tanh(self)-> "Value":
        x = self.data
        x_2 = math.exp(x * 2)
        t = (x_2 - 1) / (x_2 + 1)
        out = Value(t, (self,))

        def _backward():
            '''
                out = tanh(self)
                Dout_self = (1 - tanh(self)**2)
                一元求导法则：
                Dself = Dout_self * Dout = (1 - tanh(self)**2) * Dout
            '''
            self.grad += (1.0 - t**2) * out.grad

        out._backward = _backward
        return out

# ======= exp =======
    def exp(self)-> "Value":
        x = self.data
        y = math.exp(x)
        out = Value(y, (self,))

        def _backward():
            '''
                out = exp(self)
                Dout_self = exp(self)
                一元求导法则：
                Dself = Dout_self * Dout = exp(self) * Dout
            '''
            self.grad += y * out.grad

        out._backward = _backward
        return out
    
# ======= max =======
    def max(self, other)-> "Value":
        other = Value(other) if not isinstance(other, Value) else other
        out = Value(max(self.data, other.data), (self, other))

        def _backward():
            '''
                out = max(self, other)
                Dout_self = 1 if self.data == out.data else 0
                Dout_other = 1 if other.data == out.data else 0
                一元求导法则：
                Dself = Dout_self * Dout = 1 if self.data == out.data else 0 * Dout
                Dother = Dout_other * Dout = 1 if other.data == out.data else 0 * Dout
            '''
            self.grad += out.grad if self.data == out.data else 0
            other.grad += out.grad if other.data == out.data else 0
            
        out._backward = _backward
        return out
# ======= log =======
    def log(self, numBase: float = math.e) -> "Value":
            """
            计算以 numBase 为底的对数（默认自然对数 ln）
            公式：log_base(x) = ln(x) / ln(base) （换底公式）
            """
            # 确保底数是合法的正数（且不等于1）
            if numBase <= 0 or numBase == 1:
                raise ValueError("对数的底数必须是正数且不等于1")
            
            # 计算对数结果（使用换底公式，基于自然对数实现）
            x = self.data
            if x <= 0:
                raise ValueError("对数的输入必须是正数")
            
            # 计算 log_base(x) = ln(x) / ln(base)
            log_x = math.log(x)  # 自然对数 ln(x)
            log_base = math.log(numBase)  # 自然对数 ln(base)
            out_data = log_x / log_base  # 换底公式
            
            # 创建输出 Value，记录父节点（仅依赖 self，因为底数是常数）
            out = Value(out_data, (self,))
            
            # 定义反向传播逻辑
            def _backward():
                """
                对数的导数：d/dx [log_base(x)] = 1 / (x * ln(base))
                因此，上游梯度 * 导数 = out.grad / (x * ln(base))
                """
                derivative = 1.0 / (x * log_base)  # 导数计算
                self.grad += derivative * out.grad  # 梯度累加
            
            out._backward = _backward
            return out




# >>> 反向传播 <<<
    def backward(self):
        self.grad = 1.0
        allValueNeedToBackward = deque([self])
        # 广度优先遍历计算梯度
        while allValueNeedToBackward:
            parent = allValueNeedToBackward.popleft()
            if parent._backward is not None:
                parent._backward()
                # print(f"计算{parent}的梯度")
                for child in parent._child:
                    if child._backward is not None:
                        allValueNeedToBackward.append(child)



# 激活函数
class ActivationFunction:
    @staticmethod
    def RelU(inputs: np.ndarray) -> np.ndarray:
        # ReLU核心逻辑：max(0, value)
        def fun(value:Value):    
            return value.max(0)  # 调用Value的max方法，比较value和0
        return np.array([fun(x) for x in inputs])
        

    @staticmethod
    def softmax(values: np.ndarray) -> np.ndarray:
        # 关键修复：减去values中的最大值，避免exp溢出
        max_val = max(v.data for v in values)  # 取所有logits的最大值
        # 计算e^(x - max_val)，既不改变相对大小，又能避免溢出
        e_x = np.array([(v - max_val).exp() for v in values])  
        sum_e_x = e_x[0]
        for x in e_x[1:]:
            sum_e_x = sum_e_x + x
        return np.array([x / sum_e_x for x in e_x])


import numpy as np

class LossFunction:
    @staticmethod
    def binary_cross_entropy(y_hat: Value, y_true: float) -> Value:
        """
        二元交叉熵损失（适用于二分类）
        公式：loss = -[y_true * log(y_hat) + (1 - y_true) * log(1 - y_hat)]
        其中：
        - y_hat: 模型预测的正类概率（Value类型，范围(0,1)，通常来自Sigmoid输出）
        - y_true: 真实标签（0或1，整数）
        """
        # 防止log(0)导致数值错误（实际中可加微小epsilon，这里简化）
        term1 = y_true * y_hat.log()  # y_true * log(y_hat)
        term2 = (1 - y_true) * (1 - y_hat).log()  # (1-y_true) * log(1-y_hat)
        loss = - (term1 + term2)
        return loss

    @staticmethod
    def categorical_cross_entropy(y_hat: np.ndarray, y_true: np.ndarray) -> Value:
        """
        类别交叉熵损失（适用于多分类）
        公式：loss = -sum(y_true_i * log(y_hat_i)) 
        其中：
        - y_hat: 模型预测的概率分布（Value数组，来自Softmax输出，形状：(类别数,)）
        - y_true: 真实标签的独热编码（0/1数组，形状：(类别数,)，仅有一个1）
        """
        # 遍历每个类别，计算 y_true_i * log(y_hat_i)，再求和
        total = Value(0.0)
        for y_hat_i, y_true_i in zip(y_hat, y_true):
            if y_true_i == 1:  # 独热编码中只有真实类别为1，其余为0，可简化计算
                total += y_hat_i.log()  # 累加 log(y_hat_真实类别)
        loss = -total  # 加负号
        return loss
    
# ======== update ========
import math
from typing import List

class Optimizer: # SGD
    def __init__(self, parameters: List[Value], alpha: float = 0.05):
        self.parameters = parameters  # 待优化的参数列表（Value类型）
        self.alpha = alpha  # 学习率

    def step(self):
        """基础梯度下降（默认SGD），子类可重写以实现其他算法"""
        for param in self.parameters:
            param.data -= self.alpha * param.grad  # 参数更新

    def zero_grad(self):
        """重置所有参数的梯度（单独提取，更灵活）"""
        for param in self.parameters:
            param.grad = 0.0


class SGDMomentum(Optimizer):
    """带动量的SGD（Momentum）：模拟物理惯性，加速收敛并抑制震荡"""
    def __init__(self, parameters: List[Value], alpha: float = 0.05, momentum: float = 0.9):
        super().__init__(parameters, alpha)
        self.momentum = momentum  # 动量系数（通常取0.9）
        # 初始化每个参数的动量（与参数形状匹配，初始为0）
        self.velocities = [0.0 for _ in parameters]

    def step(self):
        for i, param in enumerate(self.parameters):
            # 动量更新：v = momentum * v + grad（累积历史梯度方向）
            self.velocities[i] = self.momentum * self.velocities[i] + param.grad
            # 参数更新：param = param - alpha * v（沿动量方向更新）
            param.data -= self.alpha * self.velocities[i]


class Adam(Optimizer):
    """Adam优化器：结合动量和自适应学习率，适用于大多数场景"""
    def __init__(self, parameters: List[Value], alpha: float = 0.001, 
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        super().__init__(parameters, alpha)
        self.beta1 = beta1  # 一阶矩（动量）系数
        self.beta2 = beta2  # 二阶矩（自适应学习率）系数
        self.eps = eps      # 防止除以0的小常数
        self.t = 0          # 迭代次数（用于偏差修正）
        # 初始化一阶矩（动量）和二阶矩（平方梯度累积）
        self.m = [0.0 for _ in parameters]  # 一阶矩（类似动量）
        self.v = [0.0 for _ in parameters]  # 二阶矩（平方梯度）

    def step(self):
        self.t += 1  # 迭代次数+1
        for i, param in enumerate(self.parameters):
            grad = param.grad
            # 累积一阶矩（动量）：m = beta1*m + (1-beta1)*grad
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            # 累积二阶矩（平方梯度）：v = beta2*v + (1-beta2)*grad^2
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            
            # 偏差修正（初期迭代的矩估计偏差）
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)  # 修正一阶矩
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)  # 修正二阶矩
            
            # 参数更新：param = param - alpha * m_hat / (sqrt(v_hat) + eps)
            param.data -= self.alpha * m_hat / (math.sqrt(v_hat) + self.eps)

def to_one_hot(label: int, num_classes: int) -> np.ndarray:
    """
    label: 整数标签（如 3）
    num_classes: 总类别数（如 10 分类任务）
    返回：独热编码数组（如 [0,0,0,1,0,0,0,0,0,0]）
    """
    one_hot = np.zeros(num_classes)
    one_hot[label] = 1.0  # 对应标签位置设为1
    return one_hot

import numpy as np
class Neuron:
    def __init__(self, in_num: int, init_type: str = "xavier", activation: str = "relu") -> None:
        """
        神经元初始化，支持不同权重初始化策略
        :param in_num: 输入特征数
        :param init_type: 初始化类型："xavier"（适用于tanh/sigmoid）、"he"（适用于relu）、"uniform"（原均匀分布）
        :param activation: 后续激活函数，用于配合初始化策略
        """
        self.in_num = in_num
        self.activation = activation
        # 根据初始化类型计算权重标准差
        if init_type == "xavier":
            # Xavier初始化：适用于tanh/sigmoid，方差=2/(in_num + out_num)，这里简化为sqrt(1/in_num)
            std = np.sqrt(1.0 / in_num)
        elif init_type == "he":
            # He初始化：适用于ReLU，方差=2/in_num
            std = np.sqrt(2.0 / in_num)
        elif init_type == "uniform":
            # 原始均匀分布（保留，作为对比）
            std = 1.0  # 后续乘(-1,1)范围
        else:
            raise ValueError(f"不支持的初始化类型：{init_type}")

        # 初始化权重（正态分布或均匀分布，基于策略）
        if init_type in ["xavier", "he"]:
            # 正态分布：均值0，标准差std
            self.weights = np.array([
                Value(np.random.randn() * std) 
                for _ in range(in_num)
            ])
        else:
            # 均匀分布：范围(-std, std)，原实现
            self.weights = np.array([
                Value(np.random.uniform(-std, std)) 
                for _ in range(in_num)
            ])

        # 偏置初始化：通常设为0（或小常数，避免初始激活值饱和）
        self.bias = Value(0.0)  # 偏置一般初始化为0更稳定

    def __call__(self, x: np.ndarray) -> Value:
        weighted_sum = np.sum(self.weights * x) + self.bias
        return weighted_sum
    
    def parameters(self):
        return list(self.weights) + [self.bias]


class Layer:
    def __init__(self, in_num: int, out_num: int, init_type: str = "xavier", activation: str = "relu") -> None:
        """
        层初始化，统一管理神经元的初始化策略
        :param in_num: 输入特征数
        :param out_num: 神经元个数（输出特征数）
        :param init_type: 权重初始化类型（传递给Neuron）
        :param activation: 该层输出使用的激活函数（用于匹配初始化策略）
        """
        self.neurons = np.array([
            Neuron(in_num, init_type=init_type, activation=activation) 
            for _ in range(out_num)
        ])
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        out = np.array([neuron(x) for neuron in self.neurons])
        return out
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
    
    def zero_grad(self):  # 修正方法名小写（PEP8规范）
        for param in self.parameters():
            param.grad = 0.0  # 重置梯度（原代码错误地重置了data，这里修正为grad）