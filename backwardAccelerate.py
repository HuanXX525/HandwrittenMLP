import math
import os
import pickle
from typing import Callable, List, Optional, Union
import numpy as np

def forNotValue(fun):
    '''如果传入的不是Value类型，则自动转为Value'''
    def wrapper(*args):
        args = [arg if isinstance(arg, Value) else Value(arg) for arg in args]
        return fun(*args)
    return wrapper

class Value():
    def __init__(self, data, child=())->None:
        self.data = np.asarray(data)
        self._child = child
        self.grad = np.zeros_like(self.data, dtype=np.float32)
        self._backward: Optional[Callable[[], None]] = None

    def __repr__(self) -> str:
        return f"Value({self.data})"

    # 加法运算
    # @forNotValue
    # def __add__(self, other)-> "Value":
    #     out = Value(self.data + other.data, (self, other))
    #     def _backward():
    #         '''
    #             out = self + other
    #             Dout_self = 1
    #             Dout_other = 1
    #             Dself = Dout * Dout_self = Dout
    #             Dother = Dout * Dout_other = Dout
    #             广播到child的shape
    #         '''
    #         self.grad += out.grad
    #         other.grad += out.grad
    #     out._backward = _backward
    #     return out

    def __radd__(self, other)-> "Value":
        return self + other
    
    def __iadd__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        self.data += other.data
        self._child = self._child + (other,)
        original_backward = self._backward

        def _backward():
            if original_backward is not None:
                original_backward()
            for child in self._child:
                child.grad += np.broadcast_to(self.grad, child.grad.shape)
        self._backward = _backward
        return self

    # 减法运算
    def __sub__(self, other)-> "Value":
        return self + (-other)

    def __rsub__(self, other)-> "Value":
        return -self + other

    # 负运算
    def __neg__(self)-> "Value":
        return self * -1

    # 乘法运算
    @forNotValue
    def __mul__(self, other) -> "Value":
        out = Value(self.data * other.data, (self, other))
        
        def _backward():
            # 计算原始梯度（可能因广播导致形状不匹配）
            self_raw_grad = other.data * out.grad
            other_raw_grad = self.data * out.grad
            
            # 处理 self 的梯度：若 self 是标量，对梯度求和
            if self.grad.ndim == 0:  # self 是标量
                self.grad += np.sum(self_raw_grad)
            else:  # self 是向量，广播适配形状
                self.grad += np.broadcast_to(self_raw_grad, self.grad.shape)
            
            # 处理 other 的梯度：若 other 是标量，对梯度求和
            if other.grad.ndim == 0:  # other 是标量
                other.grad += np.sum(other_raw_grad)
            else:  # other 是向量，广播适配形状
                other.grad += np.broadcast_to(other_raw_grad, other.grad.shape)
        
        out._backward = _backward
        return out

    @forNotValue
    def __add__(self, other)-> "Value":
        out = Value(self.data + other.data, (self, other))
        
        def _backward():
            # 加法的原始梯度就是 out.grad
            self_raw_grad = out.grad
            other_raw_grad = out.grad
            
            # 处理 self 的梯度
            if self.grad.ndim == 0:  # self 是标量
                self.grad += np.sum(self_raw_grad)
            else:
                self.grad += np.broadcast_to(self_raw_grad, self.grad.shape)
            
            # 处理 other 的梯度
            if other.grad.ndim == 0:  # other 是标量
                other.grad += np.sum(other_raw_grad)
            else:
                other.grad += np.broadcast_to(other_raw_grad, other.grad.shape)
        
        out._backward = _backward
        return out
    
    def __rmul__(self, other)-> "Value":
        return self * other

    # 指数运算和幂运算
# 在__pow__方法中修改断言
    @forNotValue
    def __pow__(self, other:float)-> "Value":
        # 允许other是data为标量的Value对象或直接是标量
        # assert isinstance(other, Value) and cp.isscalar(other.data) or isinstance(other.data, (float, int)), "指数必须是标量（int/float）"
        # 提取标量值（处理other是Value或直接标量的情况）
        # exp_val = other.data.item() if isinstance(other, Value) else other
        if isinstance(other, Value):
            exp_val = other.data.item()
        else:
            exp_val = other
        y = self.data** exp_val
        # out = Value(y, (self, other)) TODO
        out = Value(y, (self,))
        
        def _backward():
            self.grad += exp_val * (self.data **(exp_val - 1)) * out.grad
        out._backward = _backward
        return out
    # 除法运算
    def __truediv__(self, other)-> "Value":
        return self * (other ** -1)

    def __rtruediv__(self, other)-> "Value":
        return other * (self ** -1)

    # tanh
    def tanh(self)-> "Value":
        t = np.tanh(self.data)
        out = Value(t, (self,))
        def _backward():
            self.grad += (1.0 - t ** 2) * out.grad
        out._backward = _backward
        return out

    # exp
    def exp(self)-> "Value":
        y = np.exp(self.data)
        out = Value(y, (self,))
        def _backward():
            self.grad += y * out.grad
        out._backward = _backward
        return out

    # max
    def maxForSingleNormal(self, other:float)-> "Value":
        # TODO
        out_data = np.maximum(self.data, other)
        out = Value(out_data, (self, ))
        def _backward():
            self.grad += out.grad * (self.data == out_data)
            # other.grad += out.grad * (other.data == out_data)
        out._backward = _backward
        return out

    # log
    def logForSingleNormal(self, numBase: float = math.e) -> "Value":
        if numBase <= 0 or numBase == 1:
            raise ValueError("底数必须是正数且不等于1")
        x = self.data
        x = np.maximum(x, 1e-8)
        log_x = np.log(x)
        log_base = math.log(numBase) if numBase != math.e else 1.0
        out_data = log_x / log_base
        out = Value(out_data, (self,))
        def _backward():
            derivative = 1.0 / (x * log_base)
            self.grad += derivative * out.grad
        out._backward = _backward
        return out

    # 求和
    def sum(self) -> "Value":
        """
        对一维向量进行求和（仅支持一维数据）
        
        返回:
            求和结果的Value实例（标量），包含计算图和梯度传播逻辑
        """
        # 校验输入为一维
        assert self.data.ndim == 1, "sum方法仅支持一维向量"
        
        # 正向计算：一维向量求和得到标量
        sum_data = self.data.sum()  # 对一维数组求和，结果为标量
        out = Value(sum_data, (self,))  # 子节点为当前向量
        
        def _backward():
            """
            一维求和的反向传播：
            标量结果的梯度会广播到原向量的每个元素（每个元素的梯度均为结果的梯度）
            """
            # 生成与原向量同形状的梯度（全为1，再乘以out.grad实现广播）
            self.grad += np.ones_like(self.data) * out.grad
        
        out._backward = _backward
        return out

    def __getitem__(self, idx: Union[int, slice]) -> "Value":
        """
        支持索引/切片操作，返回新的Value实例（保留计算图）
        
        参数:
            idx: 整数索引或切片（如 0、1:3 等）
        返回:
            Value实例：索引/切片后的结果（标量或一维向量）
        """
        # 提取索引对应的data（确保是一维或标量）
        indexed_data = self.data[idx]
        # 创建新的Value实例，子节点为当前Value（维持计算图链路）
        out = Value(indexed_data, (self,))
        
        # 反向传播：梯度仅流向被索引的位置
        def _backward():
            # 初始化与原数据同形状的零梯度
            grad = np.zeros_like(self.grad)
            # 将输出梯度赋值到对应索引位置
            grad[idx] = out.grad
            # 累加到原节点的梯度
            self.grad += grad
        
        out._backward = _backward
        return out


    # 反向传播
    def backward(self):
        self.grad = np.ones_like(self.data)
        topo_order = []
        visited = set()
        stack = [(self, False)]
        
        while stack:
            node, processed = stack.pop()
            if processed:
                topo_order.append(node)
            else:
                if node in visited:
                    continue
                visited.add(node)
                stack.append((node, True))
                for child in reversed(node._child):
                    stack.append((child, False))
        
        for node in reversed(topo_order):
            if node._backward is not None:
                node._backward()

# 激活函数
class ActivationFunction:
    @staticmethod
    def RelU(inputs: Value) -> Value:
        return inputs.maxForSingleNormal(0)
    
    @staticmethod
    def softmax(x: Value) -> Value:
        """
        对一维Value向量计算softmax
        
        参数:
            x: Value实例，其data必须是一维数组（向量）
        返回:
            Value实例，其data为softmax结果（与输入同形状的一维向量）
        """
        # 校验输入为一维Value
        assert isinstance(x, Value), "softmax输入必须是Value实例"
        # assert x.data.ndim == 1, "softmax仅支持一维向量输入"
        
        # 步骤1：数值稳定性处理（减去最大值，避免指数溢出）
        # 提取x中的最大值（转为Value实例以兼容运算）
        max_val = Value(x.data.max())
        # 计算x - max_val（利用__sub__，保留计算图）
        x_shifted = x - max_val
        
        # 步骤2：计算指数（利用Value的exp方法，保留计算图）
        exp_x = x_shifted.exp()
        
        # 步骤3：计算指数和（利用Value的sum方法，得到标量）
        sum_exp = exp_x.sum()  # sum方法已确保一维求和，结果为标量Value
        
        # 步骤4：防止除零（给和添加极小值，不影响梯度）
        sum_exp.data = np.maximum(sum_exp.data, 1e-10)  # 避免除以0
        
        # 步骤5：计算softmax（exp_x / sum_exp，利用__truediv__保留计算图）
        softmax_out = exp_x / sum_exp
        
        return softmax_out

class LossFunction:
    @staticmethod
    def categorical_cross_entropy(y_hat: Value, label: int) -> Value:
        """
        计算类别交叉熵损失（适用于单标签分类）
        
        参数:
            y_hat: Value实例，形状为(类别数,)，表示模型输出的softmax概率分布
            label: int，真实标签的索引
        返回:
            Value实例，表示交叉熵损失值
        """
        # 校验输入合法性
        assert isinstance(y_hat, Value), "y_hat必须是Value实例"
        assert y_hat.data.ndim == 1, "y_hat必须是一维向量（softmax输出）"
        assert isinstance(label, int), "label必须是整数索引"
        assert 0 <= label < len(y_hat.data), "label索引超出y_hat范围"
        
        # 关键修复：通过Value的__getitem__提取对应位置的Value实例（而非直接取data）
        y_hat_true = y_hat[label]  # 现在y_hat_true是Value实例（标量）
        
        # 防止log(0)的数值稳定性处理（修改Value的data）
        y_hat_true.data = np.maximum(y_hat_true.data, 1e-10)
        
        # 计算交叉熵损失：-log(y_hat_true)（调用Value的log方法）
        loss = -y_hat_true.logForSingleNormal()
        
        return loss
# 优化器
class SGD:  # SGD优化器
    def __init__(self, parameters: List[Value], alpha: float = 0.05):
        self.parameters = parameters
        self.alpha = alpha

    def step(self):
        for param in self.parameters:
            param.data -= self.alpha * param.grad

    def zero_grad(self):
        for param in self.parameters:
            param.grad = np.zeros_like(param.grad)
    
    def getParameters(self):
        return self.parameters
    
    # SGD保存到文件
    def save(self, file_path: str):
        """将SGD优化器状态保存到文件"""
        # 保存核心状态：学习率+参数形状（用于校验）
        state = {
            'alpha': self.alpha,
            'param_shapes': [p.data.shape for p in self.parameters]
        }
        # 序列化到文件
        with open(file_path, 'wb') as f:
            pickle.dump(state, f)
        print(f"SGD状态已保存到 {file_path}")
    
    # SGD从文件加载
    def load(self, file_path: str):
        """从文件加载SGD优化器状态"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件 {file_path} 不存在")
        
        # 反序列化
        with open(file_path, 'rb') as f:
            state = pickle.load(f)
        
        # 校验参数形状是否匹配（防止模型结构变更）
        current_shapes = [p.data.shape for p in self.parameters]
        assert state['param_shapes'] == current_shapes, \
            f"参数形状不匹配：保存时{state['param_shapes']}，当前{current_shapes}"
        
        # 恢复状态
        self.alpha = state['alpha']
        print(f"SGD状态已从 {file_path} 加载")


class SGDMomentum(SGD):  # 带动量的SGD
    def __init__(self, parameters: List[Value], alpha: float = 0.05, momentum: float = 0.9):
        super().__init__(parameters, alpha)
        self.momentum = momentum
        self.velocities = [np.zeros_like(p.data) for p in parameters]  # 动量速度

    def step(self):
        for i, param in enumerate(self.parameters):
            self.velocities[i] = self.momentum * self.velocities[i] + param.grad
            param.data -= self.alpha * self.velocities[i]
    
    # SGDMomentum保存到文件
    def save(self, file_path: str):
        """将SGDMomentum优化器状态保存到文件"""
        # 保存核心状态：学习率+动量系数+速度数组+参数形状
        state = {
            'alpha': self.alpha,
            'momentum': self.momentum,
            'velocities': self.velocities,  # 动量速度（关键状态）
            'param_shapes': [p.data.shape for p in self.parameters]
        }
        with open(file_path, 'wb') as f:
            pickle.dump(state, f)
        print(f"SGDMomentum状态已保存到 {file_path}")
    
    # SGDMomentum从文件加载
    def load(self, file_path: str):
        """从文件加载SGDMomentum优化器状态"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件 {file_path} 不存在")
        
        with open(file_path, 'rb') as f:
            state = pickle.load(f)
        
        # 校验参数形状
        current_shapes = [p.data.shape for p in self.parameters]
        assert state['param_shapes'] == current_shapes, \
            f"参数形状不匹配：保存时{state['param_shapes']}，当前{current_shapes}"
        
        # 校验速度数组形状（与参数匹配）
        for i in range(len(state['velocities'])):
            assert state['velocities'][i].shape == current_shapes[i], \
                f"第{i}个参数的速度数组形状不匹配"
        
        # 恢复状态
        self.alpha = state['alpha']
        self.momentum = state['momentum']
        self.velocities = state['velocities']  # 恢复动量速度
        print(f"SGDMomentum状态已从 {file_path} 加载")


class Adam(SGD):  # Adam优化器
    def __init__(self, parameters: List[Value], alpha: float = 0.001, 
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        super().__init__(parameters, alpha)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0  # 迭代次数（用于偏差修正）
        self.m = [np.zeros_like(p.data) for p in parameters]  # 一阶矩
        self.v = [np.zeros_like(p.data) for p in parameters]  # 二阶矩

    def step(self):
        self.t += 1
        for i, param in enumerate(self.parameters):
            grad = param.grad
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad **2)
            m_hat = self.m[i] / (1 - self.beta1** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            param.data -= self.alpha * m_hat / (np.sqrt(v_hat) + self.eps)
    
    # Adam保存到文件
    def save(self, file_path: str):
        """将Adam优化器状态保存到文件"""
        # 保存核心状态：学习率+beta系数+迭代次数+一阶矩/二阶矩数组+参数形状
        state = {
            'alpha': self.alpha,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'eps': self.eps,
            't': self.t,  # 迭代次数（关键，影响偏差修正）
            'm': self.m,  # 一阶矩（关键状态）
            'v': self.v,  # 二阶矩（关键状态）
            'param_shapes': [p.data.shape for p in self.parameters]
        }
        with open(file_path, 'wb') as f:
            pickle.dump(state, f)
        print(f"Adam状态已保存到 {file_path}")
    
    # Adam从文件加载
    def load(self, file_path: str):
        """从文件加载Adam优化器状态"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件 {file_path} 不存在")
        
        with open(file_path, 'rb') as f:
            state = pickle.load(f)
        
        # 校验参数形状
        current_shapes = [p.data.shape for p in self.parameters]
        assert state['param_shapes'] == current_shapes, \
            f"参数形状不匹配：保存时{state['param_shapes']}，当前{current_shapes}"
        
        # 校验一阶矩/二阶矩形状（与参数匹配）
        for i in range(len(state['m'])):
            assert state['m'][i].shape == current_shapes[i], \
                f"第{i}个参数的一阶矩形状不匹配"
            assert state['v'][i].shape == current_shapes[i], \
                f"第{i}个参数的二阶矩形状不匹配"
        
        # 恢复状态
        self.alpha = state['alpha']
        self.beta1 = state['beta1']
        self.beta2 = state['beta2']
        self.eps = state['eps']
        self.t = state['t']  # 恢复迭代次数
        self.m = state['m']  # 恢复一阶矩
        self.v = state['v']  # 恢复二阶矩
        print(f"Adam状态已从 {file_path} 加载")

def to_one_hot(label: int, num_classes: int) -> np.ndarray:
    one_hot = np.zeros(num_classes, dtype=np.float32)
    one_hot[label] = 1.0
    return one_hot

# class Neuron:
#     def __init__(self, in_num: int, init_type: str = "xavier", activation: str = "relu") -> None:
#         self.in_num = in_num
#         self.activation = activation
#         if init_type == "xavier":
#             std = cp.sqrt(1.0 / in_num)
#         elif init_type == "he":
#             std = cp.sqrt(2.0 / in_num)
#         elif init_type == "uniform":
#             std = 1.0
#         else:
#             raise ValueError(f"不支持的初始化类型：{init_type}")

#         if init_type in ["xavier", "he"]:
#             self.weights = [
#                 Value(cp.random.randn() * std)
#                 for _ in range(in_num)
#             ]
#         else:
#             self.weights = [
#                 Value(cp.random.uniform(-std, std))
#                 for _ in range(in_num)
#             ]
#         self.bias = Value(0.0)

#     def __call__(self, x: cp.ndarray) -> Value:
#         weighted_sum = sum(w * x_i for w, x_i in zip(self.weights, x)) + self.bias
#         return weighted_sum
    
#     def parameters(self):
#         return self.weights + [self.bias]

class Layer:
    def __init__(self, in_num: int, out_num: int, init_type: str = "xavier") -> None:
        """
        全连接层（权重W为一维Value数组，每个元素对应一个输出神经元的权重向量）
        
        参数:
            in_num: 输入特征数（每个权重向量的长度）
            out_num: 输出特征数（权重数组的长度）
            init_type: 权重初始化方式（xavier/normal/uniform）
        """
        self.in_num = in_num
        self.out_num = out_num
        
        # 初始化权重：W是一维Value的数组，形状为(out_num,)，每个元素是形状为(in_num,)的Value
        self.W = []
        for _ in range(out_num):
            if init_type == "xavier":
                bound = np.sqrt(6.0 / (in_num + out_num))
                w_data = np.random.uniform(-bound, bound, in_num)  # 一维权重向量
            elif init_type == "normal":
                w_data = np.random.normal(0, 0.01, in_num)
            elif init_type == "uniform":
                w_data = np.random.uniform(-0.05, 0.05, in_num)
            else:
                raise ValueError(f"不支持的初始化方式：{init_type}")
            self.W.append(Value(w_data))  # 每个权重向量都是一维Value
        
        # 初始化偏置：b是一维Value数组，每个元素是标量（形状为()的Value）
        self.b = [Value(0.0) for _ in range(out_num)]  # 每个偏置都是标量Value

    def __call__(self, x: Value) -> Value:
        """
        前向传播（输入x是一维Value，输出是一维Value）
        
        参数:
            x: 一维Value实例，形状为(in_num,)（单样本特征）
        返回:
            一维Value实例，形状为(out_num,)（单样本输出）
        """
        # 校验输入合法性（x必须是一维Value，长度匹配输入特征数）
        assert isinstance(x, Value), "输入x必须是Value实例"
        # assert x.data.ndim == 1, "输入x必须是一维向量"
        # assert len(x.data) == self.in_num, f"输入长度不符：需{self.in_num}，实际{len(x.data)}"
        
        # 计算每个输出神经元的线性结果（x·w_i + b_i）
        linear_outs = []
        for i in range(self.out_num):
            w_i = self.W[i]  # 第i个输出神经元的权重向量（一维Value）
            b_i = self.b[i]  # 第i个输出神经元的偏置（标量Value）
            
            # 计算 x 与 w_i 的点积：x1*w1 + x2*w2 + ... + xn*wn
            xw = x * w_i  # 元素乘法（一维Value，形状(in_num,)）
            sum_xw = xw.sum()  # 求和得到标量Value
            
            # 加上偏置，得到第i个输出
            out_i = sum_xw + b_i  # 标量Value
            linear_outs.append(out_i)
        
        # 将所有输出合并为一个一维Value（形状(out_num,)）
        # 1. 提取所有输出的数据，拼接成一维数组
        out_data = np.array([v.data for v in linear_outs])
        # 2. 收集所有依赖的子节点（确保计算图完整）
        children = {x}
        children.update(self.W)  # 所有权重向量
        children.update(self.b)  # 所有偏置
        for out_i in linear_outs:
            children.update(out_i._child)  # 中间计算节点
        # 3. 创建最终输出的Value
        out = Value(out_data, tuple(children))
        
        # 定义反向传播：将总梯度分配到每个输出神经元
        def _backward():
            for i in range(self.out_num):
                # 当前输出神经元的梯度（从总梯度中提取对应位置）
                grad_i = out.grad[i]
                # 将梯度传递给中间输出out_i
                linear_outs[i].grad += grad_i
                # 触发out_i的反向传播（更新x、w_i、b_i的梯度）
                if linear_outs[i]._backward is not None:
                    linear_outs[i]._backward()
        
        out._backward = _backward
        return out

    def parameters(self) -> List[Value]:
        """返回所有可训练参数（权重数组+偏置数组）"""
        return self.W + self.b  # 合并权重和偏置的一维Value列表

    def zero_grad(self) -> None:
        """重置所有参数的梯度为0"""
        for param in self.parameters():
            param.grad = np.zeros_like(param.grad, dtype=np.float32)

    def save_model(self, filename: Optional[str] = None) -> None:
        """保存模型参数（仅存储数据，兼容一维Value）"""
        filename = filename or "layer_model.pkl"
        # 存储权重和偏置的数据（W是out_num个in_num维向量，b是out_num个标量）
        model_data = {
            "in_num": self.in_num,
            "out_num": self.out_num,
            "W_data": [w.data for w in self.W],  # 列表：每个元素是(in_num,)数组
            "b_data": [b.data for b in self.b]   # 列表：每个元素是标量
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)

    @staticmethod
    def load_model(filename: Optional[str] = None) -> "Layer":
        """加载模型并重建Layer实例"""
        filename = filename or "layer_model.pkl"
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        # 重建层（无需指定初始化方式，直接恢复数据）
        layer = Layer(
            in_num=model_data["in_num"],
            out_num=model_data["out_num"]
        )
        # 恢复权重数据（每个W[i]是一维Value）
        for i in range(layer.out_num):
            layer.W[i].data = model_data["W_data"][i]
        # 恢复偏置数据（每个b[i]是标量Value）
        for i in range(layer.out_num):
            layer.b[i].data = model_data["b_data"][i]
        return layer