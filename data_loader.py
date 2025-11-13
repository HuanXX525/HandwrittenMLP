import numpy as np
import struct
import os


def load_mnist_images(file_path, flatten=False):
    """
    读取MNIST图像文件
    
    Args:
        file_path (str): 图像文件路径
        flatten (bool): 是否将图像展平为二维数组(N, pixels)
        
    Returns:
        np.ndarray: 图像数据，形状为(N, 28, 28)或(N, 784)，像素值范围[0, 1]
    """
    with open(file_path, 'rb') as f:
        # 读取文件头信息
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        # 验证魔数
        if magic != 2051:
            raise ValueError(f'Invalid magic number: {magic}')
        
        # 读取图像数据
        images = np.fromfile(f, dtype=np.uint8)
        # 重塑为(N, 28, 28)形状
        images = images.reshape(num_images, rows, cols)
        
        # 将图像数据归一化到[0, 1]范围
        images = images.astype(np.float32) / 255.0
        
        # 如果需要展平为二维数组
        if flatten:
            images = images.reshape(num_images, rows * cols)
        
    return images


def load_mnist_labels(file_path):
    """
    读取MNIST标签文件
    
    Args:
        file_path (str): 标签文件路径
        
    Returns:
        np.ndarray: 标签数据，形状为(N,)
    """
    with open(file_path, 'rb') as f:
        # 读取文件头信息
        magic, num_labels = struct.unpack('>II', f.read(8))
        # 验证魔数
        if magic != 2049:
            raise ValueError(f'Invalid magic number: {magic}')
        
        # 读取标签数据
        labels = np.fromfile(f, dtype=np.uint8)
        
    return labels


def load_mnist_dataset(data_dir, flatten=False):
    """
    加载MNIST数据集
    
    Args:
        data_dir (str): 数据目录路径
        flatten (bool): 是否将图像展平为二维数组(N, pixels)
        
    Returns:
        tuple: (train_images, train_labels, test_images, test_labels)
    """
    # 加载训练集
    train_images_path = os.path.join(data_dir, 'train-images-idx3-ubyte')
    train_labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte')
    train_images = load_mnist_images(train_images_path, flatten=flatten)
    train_labels = load_mnist_labels(train_labels_path)
    
    # 加载测试集
    test_images_path = os.path.join(data_dir, 't10k-images.idx3-ubyte')
    test_labels_path = os.path.join(data_dir, 't10k-labels.idx1-ubyte')
    test_images = load_mnist_images(test_images_path, flatten=flatten)
    test_labels = load_mnist_labels(test_labels_path)
    
    return train_images, train_labels, test_images, test_labels