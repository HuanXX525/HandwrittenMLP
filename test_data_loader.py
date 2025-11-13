import numpy as np
from data_loader import load_mnist_dataset


def test_data_loader():
    """测试数据加载器"""
    # 加载数据集（默认3D格式）
    train_images, train_labels, test_images, test_labels = load_mnist_dataset('./data')
    
    # 检查数据形状
    assert train_images.shape == (60000, 28, 28), f"训练图像形状不正确: {train_images.shape}"
    assert train_labels.shape == (60000,), f"训练标签形状不正确: {train_labels.shape}"
    assert test_images.shape == (10000, 28, 28), f"测试图像形状不正确: {test_images.shape}"
    assert test_labels.shape == (10000,), f"测试标签形状不正确: {test_labels.shape}"
    
    # 检查数据类型
    assert train_images.dtype == np.uint8, f"训练图像数据类型不正确: {train_images.dtype}"
    assert train_labels.dtype == np.uint8, f"训练标签数据类型不正确: {train_labels.dtype}"
    assert test_images.dtype == np.uint8, f"测试图像数据类型不正确: {test_images.dtype}"
    assert test_labels.dtype == np.uint8, f"测试标签数据类型不正确: {test_labels.dtype}"
    
    # 检查像素值范围
    assert train_images.min() >= 0 and train_images.max() <= 255, "训练图像像素值超出范围"
    assert test_images.min() >= 0 and test_images.max() <= 255, "测试图像像素值超出范围"
    
    # 检查标签值范围
    assert train_labels.min() >= 0 and train_labels.max() <= 9, "训练标签值超出范围0-9"
    assert test_labels.min() >= 0 and test_labels.max() <= 9, "测试标签值超出范围0-9"
    
    print("3D格式测试通过!")
    print(f"训练集: {train_images.shape[0]} 图像, 标签")
    print(f"测试集: {test_images.shape[0]} 图像, 标签")
    print(f"图像形状: {train_images.shape[1:]}")
    print(f"标签范围: {np.unique(train_labels)}")
    
    # 测试展平格式
    train_images_flat, train_labels_flat, test_images_flat, test_labels_flat = load_mnist_dataset('./data', flatten=True)
    
    # 检查展平后的数据形状
    assert train_images_flat.shape == (60000, 784), f"训练图像展平形状不正确: {train_images_flat.shape}"
    assert train_labels_flat.shape == (60000,), f"训练标签形状不正确: {train_labels_flat.shape}"
    assert test_images_flat.shape == (10000, 784), f"测试图像展平形状不正确: {test_images_flat.shape}"
    assert test_labels_flat.shape == (10000,), f"测试标签形状不正确: {test_labels_flat.shape}"
    
    # 检查展平后的数据类型
    assert train_images_flat.dtype == np.uint8, f"训练图像展平数据类型不正确: {train_images_flat.dtype}"
    assert train_labels_flat.dtype == np.uint8, f"训练标签数据类型不正确: {train_labels_flat.dtype}"
    assert test_images_flat.dtype == np.uint8, f"测试图像展平数据类型不正确: {test_images_flat.dtype}"
    assert test_labels_flat.dtype == np.uint8, f"测试标签数据类型不正确: {test_labels_flat.dtype}"
    
    # 检查展平后的像素值范围
    assert train_images_flat.min() >= 0 and train_images_flat.max() <= 255, "训练图像展平像素值超出范围"
    assert test_images_flat.min() >= 0 and test_images_flat.max() <= 255, "测试图像展平像素值超出范围"
    
    # 检查展平后的标签值范围
    assert train_labels_flat.min() >= 0 and train_labels_flat.max() <= 9, "训练标签值超出范围0-9"
    assert test_labels_flat.min() >= 0 and test_labels_flat.max() <= 9, "测试标签值超出范围0-9"
    
    print("\n2D展平格式测试通过!")
    print(f"训练集: {train_images_flat.shape[0]} 图像, 标签")
    print(f"测试集: {test_images_flat.shape[0]} 图像, 标签")
    print(f"图像形状: {train_images_flat.shape[1:]}")
    print(f"标签范围: {np.unique(train_labels_flat)}")


if __name__ == "__main__":
    test_data_loader()