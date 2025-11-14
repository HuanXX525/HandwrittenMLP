from backward import ActivationFunction, LossFunction, SGD, to_one_hot, Layer, SGDMomentum, Adam
from data_loader import load_mnist_dataset
import numpy as np
numInputs, numOutputs, numHiddens = 784, 10, 256 # 定义模型参数

L1 = Layer(numInputs, numHiddens)
L2 = Layer(numHiddens, numOutputs)

opti = SGD(L1.parameters()+L2.parameters(), 0.01)

train_images, train_labels, test_images, test_labels = load_mnist_dataset('./data', flatten=True)
train_images = np.array(train_images)
# train_images = np.array([[0, 0, 4], [5, 0, 1], [1, 5, 0]])

# train_labels = np.array([2, 0, 1])



lossL = [0] * 10
for epoch in range(11):
    print(f"Epoch: {epoch}")
    for imgIdx in range(100):

        img = train_images[imgIdx]

        label = train_labels[imgIdx]
        # 前向传播
        a1 = L1(img)
        # print(f"a1: {a1}")
        a2 = L2(ActivationFunction.RelU(a1))
        # print(f"a2: {a2}")
        yHat = ActivationFunction.softmax(a2)
        # print(f"yHat: {yHat}")
        # print(oh)
        loss = LossFunction.categorical_cross_entropy(yHat, to_one_hot(label, 10))
        diff = loss.data - lossL[label]
        lossL[label] = loss.data
        formatted_losses = [f"{loss:6.2f}" for loss in lossL]
        print(f"losses{label}:{diff}: {', '.join(formatted_losses)}")
        print(f"avarage loss: {np.mean(lossL)}")
        # print("Forward pass completed")
        opti.zero_grad()
        loss.backward()
        opti.step()

L1.save_model('L1.pkl')
L2.save_model('L2.pkl')