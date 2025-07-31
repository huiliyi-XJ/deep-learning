import torch
import torch.nn as nn

def explain_linear_layers():
    print("=== 全连接层详细解释 ===\n")
    
    # 模拟输入数据
    batch_size = 4
    input_features = torch.randn(batch_size, 80)
    print(f"输入数据形状: {input_features.shape}")
    print(f"每个样本有80个特征\n")
    
    # 创建三个全连接层
    linear1 = nn.Linear(80, 40)
    linear2 = nn.Linear(40, 20)
    linear3 = nn.Linear(20, 10)
    
    print("=== 第一层全连接: 80 → 40 ===")
    print("输入: 80个特征")
    print("输出: 40个特征")
    print("权重矩阵: (80, 40)")
    print("偏置向量: (40,)")
    
    x = linear1(input_features)
    print(f"输出形状: {x.shape}")
    print(f"每个样本从80维压缩到40维\n")
    
    print("=== 第二层全连接: 40 → 20 ===")
    print("输入: 40个特征")
    print("输出: 20个特征")
    print("权重矩阵: (40, 20)")
    print("偏置向量: (20,)")
    
    x = linear2(x)
    print(f"输出形状: {x.shape}")
    print(f"每个样本从40维压缩到20维\n")
    
    print("=== 第三层全连接: 20 → 10 ===")
    print("输入: 20个特征")
    print("输出: 10个特征（对应10个数字类别）")
    print("权重矩阵: (20, 10)")
    print("偏置向量: (10,)")
    
    x = linear3(x)
    print(f"输出形状: {x.shape}")
    print(f"每个样本从20维压缩到10维\n")
    
    print("=== 最终输出 ===")
    print("每个样本输出10个数字，代表对0-9每个数字的预测概率")
    print("通过softmax函数转换为概率分布")

def explain_linear_transformation():
    print("\n=== 线性变换的数学原理 ===")
    
    # 手动实现线性变换
    input_size = 80
    output_size = 40
    
    # 创建权重和偏置
    W = torch.randn(input_size, output_size)
    b = torch.randn(output_size)
    
    # 模拟输入
    x = torch.randn(1, input_size)
    
    print(f"输入向量 x: {x.shape}")
    print(f"权重矩阵 W: {W.shape}")
    print(f"偏置向量 b: {b.shape}")
    
    # 线性变换
    y = torch.matmul(x, W) + b
    print(f"输出向量 y: {y.shape}")
    
    print("\n数学公式: y = Wx + b")
    print("这是一个标准的线性变换！")

def explain_feature_compression():
    print("\n=== 特征压缩的含义 ===")
    print("80 → 40 → 20 → 10 的过程：")
    print("1. 80维: 从卷积层提取的原始特征")
    print("2. 40维: 第一次特征压缩，保留主要特征")
    print("3. 20维: 第二次特征压缩，进一步抽象")
    print("4. 10维: 最终分类特征，对应10个数字类别")
    
    print("\n为什么需要压缩？")
    print("- 减少参数数量，防止过拟合")
    print("- 提取更抽象的特征")
    print("- 逐步将空间特征转换为分类特征")

def show_parameter_count():
    print("\n=== 参数数量统计 ===")
    
    # 计算每层的参数数量
    linear1_params = 80 * 40 + 40  # 权重 + 偏置
    linear2_params = 40 * 20 + 20
    linear3_params = 20 * 10 + 10
    
    print(f"第一层参数: {linear1_params} (80×40 + 40)")
    print(f"第二层参数: {linear2_params} (40×20 + 20)")
    print(f"第三层参数: {linear3_params} (20×10 + 10)")
    print(f"总参数: {linear1_params + linear2_params + linear3_params}")

if __name__ == "__main__":
    explain_linear_layers()
    explain_linear_transformation()
    explain_feature_compression()
    show_parameter_count() 