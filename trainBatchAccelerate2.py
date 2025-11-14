from datetime import datetime
from enum import Enum
from os import path

from matplotlib import pyplot as plt
from backwardAccelerate import ActivationFunction, LossFunction, SGD, to_one_hot, Layer, SGDMomentum, Adam, Value
from data_loader import load_mnist_dataset
import numpy as np
import argparse
import logging



class OptiName(Enum):
    SGD = "SGD"
    Momentum = "Momentum"
    Adam = "Adam"
# =============== 命令行参数解析 ===============
parser = argparse.ArgumentParser(description='训练并测试使用不同优化器的神经网络，若能找到pkl文件则直接读取并测试，否则重新训练')
parser.add_argument('-o', '--optimizer', 
                    type=str,
                    choices=[opti.value for opti in OptiName],  # 限制输入选项
                    help=f'优化器类型 {OptiName.SGD.value}|{OptiName.Momentum.value}|{OptiName.Adam.value}', 
                    default=OptiName.SGD.value)
parser.add_argument('-e', '--epoch', type=int, help='训练次数', default=10)
parser.add_argument('-lr', '--learningRate', type=float, help='学习率', default=0.001)
parser.add_argument('-c', '--continueTrain', type=bool, help='继续训练', default=False)
parser.add_argument('-n', '--numOfTrain', type=int, help='训练数据量', default=None)
parser.add_argument('-b', "--batchSize", type=int, help='批量大小', default=32)

args = parser.parse_args()
# =============== 命令行参数解析 ===============

# =============== 日志配置 ===============
logging.basicConfig(
    filename=f'{args.optimizer}.log',  # 日志文件名
    level=logging.INFO,  # 日志级别（INFO及以上会被记录）
    format='%(message)s'  # 日志格式
)
# =============== 日志配置 ===============

# =============== 训练测试流程 ===============
def train(optiName, lr, epoches, numOfTrain=None):
    logging.info(f"============= TRAIN LOG =============")
    logging.info(f"TIMESTAMP: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Optimizer: {optiName}")
    logging.info(f"Learning Rate: {lr}")
    logging.info(f"Epochs: {epoches}")
    logging.info(f"Contine Train: {args.continueTrain}")
    # 模型参数
    numInputs, numOutputs, numHiddens = 784, 10, 256
    batch_size = args.batchSize


    # 初始化模型和优化器
    if not args.continueTrain:
        L1 = Layer(numInputs, numHiddens)
        L2 = Layer(numHiddens, numOutputs)
    else:
        L1 = Layer.load_model(f'{optiName}L1.pkl')
        L2 = Layer.load_model(f'{optiName}L2.pkl')

    if (optiName == OptiName.SGD):
        opti = SGD(L1.parameters() + L2.parameters(), lr)
    elif (optiName == OptiName.Momentum):
        opti = SGDMomentum(L1.parameters() + L2.parameters(), lr)
    else:
        opti = Adam(L1.parameters() + L2.parameters(), lr)
    if args.continueTrain:
        opti.load(f'{optiName}Opti.pkl')
    # 加载训练数据
    train_images, train_labels = load_mnist_dataset('./data', flatten=True)
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)

    # 记录每个类别的损失（用于监控）
    lossL = [0.0] * 10
    # 记录每个批次的损失（用于画图）
    lossL_batch = []

    totle = len(train_images) if numOfTrain is None else numOfTrain
    logging.info(f"Num of sample: {totle}")

    # 训练循环（批量版）
    for epoch in range(epoches):
        print(f"\nEpoch: {epoch}")
        total_batches = totle // batch_size  # 总批次数
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
            
            # 记录打印批量信息
            lossL_batch.append(batch_avg_loss.data)
            formatted_losses = [f"{loss:6.2f}" for loss in lossL]
            print(f"Batch {batch_idx}/{total_batches} | Avg Loss: {batch_avg_loss.data:.4f} | Class Losses: {', '.join(formatted_losses)}")
        
        # 打印epoch信息
        print(f"Epoch {epoch} | Total Avg Loss: {epoch_loss / total_batches:.4f}\n")
        logging.info(f"Epoch {epoch} | Total Avg Loss: {epoch_loss / total_batches:.4f}\n")
        if (epoch % 5 == 0 or epoch == epoches - 1):
            L1.save_model(f'{optiName}_{epoch}L1.pkl')
            L2.save_model(f'{optiName}_{epoch}L2.pkl')
    # 保存模型保存损失曲线
    L1.save_model(f'{optiName}L1.pkl')
    L2.save_model(f'{optiName}L2.pkl')
    opti.save(f'{optiName}Opti.pkl')
    # plt.plot(lossL_batch)
    # plt.title(f'{optiName} Loss Curve')
    # plt.savefig(f'{optiName}LossCurve.png')
    plt.figure(figsize=(12, 6))
    plt.plot(lossL_batch, label='Batch Loss')
    plt.xlabel('Batch Index')
    plt.ylabel('Loss Value')
    plt.title(f'{optiName} Loss Curve (LR={lr}, Epochs={epoches})')
    plt.grid(alpha=0.3)
    plt.legend()
    # 添加epoch分隔线
    for i in range(1, epoches):
        plt.axvline(x=i*(totle//batch_size), color='r', linestyle='--', alpha=0.3)
    plt.savefig(f'{optiName}LossCurve.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved model and loss curve as {optiName}L1.pkl and {optiName}L2.pkl and {optiName}LossCurve.png")
    logging.info(f"============= TRAIN LOG =============")


def test(optiName):
    logging.info(f"============= TEST LOG =============")
    logging.info(f"TIMESTAMP: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Optimizer: {optiName}")

    # 加载模型
    L1 = Layer.load_model(f'{optiName}L1.pkl')
    L2 = Layer.load_model(f'{optiName}L2.pkl')

    # 加载测试数据
    test_images, test_labels = load_mnist_dataset('./data', True, "test")

    correct = 0
    total = 0
    accuracy = 0
    for i, image in enumerate(test_images):
        x = Value(image)
        a1 = L1(x)
        a2 = L2(ActivationFunction.RelU(a1))
        yHat = ActivationFunction.softmax(a2)
        pred = np.argmax(yHat.data)
        trueLabel = test_labels[i]
        if (pred == trueLabel):
            correct += 1
        total += 1
        if i % 100 == 0:
            accuracy = correct / total
            print("Test Accuracy:", accuracy)
    accuracy = correct / total
    logging.info(f"Test Accuracy: {accuracy}")

# =============== 训练测试流程 ===============

if __name__ == '__main__':
    optimizer = args.optimizer
    if path.exists(f'{optimizer}L1.pkl') and path.exists(f'{optimizer}L2.pkl') and not args.continueTrain:
        test(optimizer)
    else:
        train(optiName=optimizer, epoches=args.epoch, lr=args.learningRate, numOfTrain=args.numOfTrain)
