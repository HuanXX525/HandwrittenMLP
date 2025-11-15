from backward import ActivationFunction, LossFunction, to_one_hot, Layer, Adam
import numpy as np
numInputs, numOutputs, numHiddens = 100, 10, 50 # 定义模型参数
from drawhuge import *

L1 = Layer(numInputs, numHiddens)
L2 = Layer(numHiddens, numOutputs)

opti = Adam(L1.parameters()+L2.parameters(), 0.01)
# 前向传播
a1 = L1(np.random.randn(numInputs).reshape((1,-1)).astype("float32"))
a2 = L2(ActivationFunction.RelU(a1))
yHat = ActivationFunction.softmax(a2)
loss = LossFunction.categorical_cross_entropy(yHat, to_one_hot(0, numOutputs))
opti.zero_grad()
loss.backward()
draw_dot(loss, filename="MLPCG")
