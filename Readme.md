# 手动实现与优化多层感知机（MLP）

- 实验目的：深入理解前向传播、反向传播机制以及各种优化算法（SGD, Momentum, Adam）的工作原理。
- 实验内容：不使用深度学习框架的高层API（如PyTorch的nn.Module），仅使用NumPy，可以使用自动微分库（如Autograd）但必须理解反向传播原理。在数据集上训练模型，并比较不同优化器的收敛速度、最终精度和训练稳定性。
- 实验数据集：MNIST
- 评价指标：测试集准确率、训练损失曲线、收敛迭代次数。
- 相关经典论文：Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors.Nature.
