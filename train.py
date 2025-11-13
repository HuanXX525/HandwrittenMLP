from backward import ActivationFunction, LossFunction, Optimizer, to_one_hot, Layer
from data_loader import load_mnist_dataset

numInputs, numOutputs, numHiddens = 784, 10, 256 # 定义模型参数

L1 = Layer(numInputs, numHiddens)
L2 = Layer(numHiddens, numOutputs)

opti = Optimizer(L1.parameters()+L2.parameters(), 0.00001) 

train_images, train_labels, test_images, test_labels = load_mnist_dataset('./data', flatten=True)



lossL = [0] * 10
for imgIdx in range(train_images.shape[0]):
    img = 1.0 * train_images[imgIdx]
    label = train_labels[imgIdx]
    # 前向传播
    a1 = L1(img)
    a2 = L2(ActivationFunction.RelU(a1))
    yHat = ActivationFunction.softmax(a2)
    
    loss = LossFunction.categorical_cross_entropy(yHat, to_one_hot(label, 10))
    diff = loss.data - lossL[label]
    lossL[label] = loss.data
    formatted_losses = [f"{loss:6.2f}" for loss in lossL]
    print(f"losses{label}:{diff}: {', '.join(formatted_losses)}")
    opti.zero_grad()
    loss.backward()
    opti.step()
