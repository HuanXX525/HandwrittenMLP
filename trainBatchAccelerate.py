from backwardAccelerate import ActivationFunction, LossFunction, Optimizer, to_one_hot, Layer, SGDMomentum, Adam, Value
from data_loader import load_mnist_dataset
import numpy as np

# 模型参数
numInputs, numOutputs, numHiddens = 784, 10, 256
batch_size = 32  # 设置批量大小（关键新增）

# 初始化模型和优化器
L1 = Layer(numInputs, numHiddens)
L2 = Layer(numHiddens, numOutputs)
opti = Adam(L1.parameters() + L2.parameters(), alpha=0.001)  # 建议用Adam，批量训练更稳定

# 加载数据
train_images, train_labels, test_images, test_labels = load_mnist_dataset('./data', flatten=True)
train_images = np.array(train_images)
train_labels = np.array(train_labels)

# 记录每个类别的损失（用于监控）
lossL = [0.0] * 10

# 训练循环（批量版）
for epoch in range(11):
    print(f"\nEpoch: {epoch}")
    total_batches = len(train_images) // batch_size  # 总批次数
    epoch_loss = 0.0  # 记录整个epoch的总损失
    
    for batch_idx in range(total_batches):
        # 1. 提取当前批量的数据
        start = batch_idx * batch_size
        end = start + batch_size
        batch_images = train_images[start:end]
        batch_labels = train_labels[start:end]
        
        # 2. 批量内累加损失和梯度（不清零，累积梯度）
        batch_total_loss = Value(0.0)  # 累加当前批量的损失
        opti.zero_grad()  # 批量开始前清零梯度（关键：确保每个批量梯度独立）
        
        for img, label in zip(batch_images, batch_labels):
            # 前向传播
            a1 = L1(Value(img))
            a2 = L2(ActivationFunction.RelU(a1))
            yHat = ActivationFunction.softmax(a2)
            # oh = to_one_hot(label, 10)
            loss = LossFunction.categorical_cross_entropy(yHat, int(label))
            
            # 累加单样本损失（用于计算批量平均损失）
            batch_total_loss += loss
            
            # 记录每个类别的损失（监控用）
            lossL[label] = 0.9 * lossL[label] + 0.1 * loss.data  # 平滑处理，避免波动过大
        
        # 3. 批量损失取平均（关键：梯度对应平均损失）
        batch_avg_loss = batch_total_loss / batch_size
        epoch_loss += batch_avg_loss.data
        
        # 4. 反向传播（批量梯度自动累加）
        batch_avg_loss.backward()
        
        # 5. 批量结束后更新一次参数
        opti.step()
        
        # 打印批量信息
        formatted_losses = [f"{loss:6.2f}" for loss in lossL]
        print(f"Batch {batch_idx}/{total_batches} | Avg Loss: {batch_avg_loss.data:.4f} | Class Losses: {', '.join(formatted_losses)}")
    
    # 打印epoch信息
    print(f"Epoch {epoch} | Total Avg Loss: {epoch_loss / total_batches:.4f}\n")

# 保存模型
L1.save_model('L1.pkl')
L2.save_model('L2.pkl')