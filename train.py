from backward import ActivationFunction, LossFunction, to_one_hot, Layer, Adam
import numpy as np
numInputs, numOutputs, numHiddens = 20, 10, 15 # 定义模型参数
from drawhuge import *

L1 = Layer(numInputs, numHiddens)
L2 = Layer(numHiddens, numOutputs)

opti = Adam(L1.parameters()+L2.parameters(), 0.01)
# 前向传播
a1 = L1(np.random.randn(20).reshape((1,-1)).astype("float64"))
a2 = L2(ActivationFunction.RelU(a1))
yHat = ActivationFunction.softmax(a2)
loss = LossFunction.categorical_cross_entropy(yHat, to_one_hot(0, 10))
opti.zero_grad()
loss.backward()
draw_dot(loss, filename="huge")
