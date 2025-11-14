from backwardAccelerate import ActivationFunction, LossFunction, Optimizer, to_one_hot, Layer, Value
import numpy as np

def simple_test():
    """
    简单测试神经网络框架的正确性
    使用一个小型数据集进行训练和验证
    """
    print("开始简单测试...")
    
    # 创建小型测试数据 (4个样本, 每个样本2个特征)
    train_X = np.array([
        [0.1, 0.9],  # 样本1 - 类别0
        [0.2, 0.9],  # 样本2 - 类别0
        [0.8, 0.1],  # 样本3 - 类别1
        [0.9, 0.2]   # 样本4 - 类别1
    ])
    
    # 对应标签 (独热编码)
    train_y = np.array([
        0,  # 样本1标签: 类别0
        0,  # 样本2标签: 类别0
        1,  # 样本3标签: 类别1
        1   # 样本4标签: 类别1
    ])
    
    # 定义网络结构 (2输入, 8隐藏单元, 2输出)
    input_size = 2
    hidden_size = 30
    output_size = 2
    
    # 创建网络层
    L1 = Layer(input_size, hidden_size, init_type="xavier")
    L2 = Layer(hidden_size, output_size, init_type="xavier")
    
    # 创建优化器 (使用较小的学习率)
    optimizer = Optimizer(L1.parameters() + L2.parameters(), alpha=0.01)
    # print(optimizer.getParameters())
    # print(L1.parameters())
    # 训练几个epoch
    num_epochs = 500
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        
        # 遍历所有训练样本
        for i in range(len(train_X)):
            # 前向传播
            x = Value(train_X[i])
            a1 = L1(x)
            a2 = L2(ActivationFunction.RelU(a1))
            y_hat = ActivationFunction.softmax(a2)
            
            # 计算损失
            loss = LossFunction.categorical_cross_entropy(y_hat, int(train_y[i]))
            total_loss += loss.data
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # 每50个epoch打印一次损失
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_X):.6f}")
    
    # 测试最终结果
    # print(L1.parameters())

    print("\n最终预测结果:")
    correct_predictions = 0
    for i in range(len(train_X)):
        x = Value(train_X[i])
        a1 = L1(x)
        a2 = L2(ActivationFunction.RelU(a1))
        y_hat = ActivationFunction.softmax(a2)
        
        # 打印预测结果和真实标签
        # predicted_class = np.argmax([v.data for v in y_hat.data])
        predicted_class = np.argmax(y_hat.data)
        true_class = train_y[i]
        
        # 统计正确预测数
        if predicted_class == true_class:
            correct_predictions += 1
            
        print(f"样本 {i+1}: 预测类别={predicted_class}, 真实类别={true_class}")
        
        # 打印概率分布
        probs = y_hat.data
        print(f"  预测概率分布: {[f'{p:.4f}' for p in probs]}")
        print(f"  真实标签: {train_y[i]}")
    
    print(f"\n训练集准确率: {correct_predictions/len(train_X)*100:.2f}%")
    print("测试完成!")

if __name__ == "__main__":
    simple_test()