import torch
import torch.nn as nn


def demonstrate_feature_compression():
    print("=== 特征压缩过程演示 ===\n")

    # 模拟一个样本的特征
    sample_features = torch.randn(1, 80)
    print(f"原始特征 (80维): {sample_features.shape}")
    print(f"特征值范围: [{sample_features.min():.3f}, {sample_features.max():.3f}]")

    # 创建全连接层
    linear1 = nn.Linear(80, 40)
    linear2 = nn.Linear(40, 20)
    linear3 = nn.Linear(20, 10)

    # 第一层压缩
    features_40 = linear1(sample_features)
    print(f"\n第一层压缩后 (40维): {features_40.shape}")
    print(f"特征值范围: [{features_40.min():.3f}, {features_40.max():.3f}]")

    # 第二层压缩
    features_20 = linear2(features_40)
    print(f"\n第二层压缩后 (20维): {features_20.shape}")
    print(f"特征值范围: [{features_20.min():.3f}, {features_20.max():.3f}]")

    # 第三层压缩
    features_10 = linear3(features_20)
    print(f"\n第三层压缩后 (10维): {features_10.shape}")
    print(f"特征值范围: [{features_10.min():.3f}, {features_10.max():.3f}]")

    # 显示具体的数值
    print(f"\n=== 具体数值对比 ===")
    print(f"原始特征 (前5个): {sample_features[0, :5]}")
    print(f"40维特征 (前5个): {features_40[0, :5]}")
    print(f"20维特征 (前5个): {features_20[0, :5]}")
    print(f"10维特征 (全部): {features_10[0]}")

    # 应用softmax得到概率
    probabilities = torch.softmax(features_10, dim=1)
    print(f"\n最终概率分布: {probabilities[0]}")
    print(f"概率和: {probabilities.sum():.6f}")

    # 预测结果
    predicted_class = torch.argmax(probabilities, dim=1)
    print(f"预测的数字: {predicted_class.item()}")


def explain_why_linear():
    print("\n=== 为什么全连接层是线性的？ ===")
    print("1. 数学定义：y = Wx + b 是线性函数")
    print("2. 没有激活函数：纯线性变换")
    print("3. 可组合性：多个线性层可以合并")
    print("4. 计算效率：线性变换计算快速")
    print("5. 梯度稳定：线性变换梯度稳定")


def show_parameter_details():
    print("\n=== 参数详情 ===")

    # 创建网络
    model = nn.Sequential(nn.Linear(80, 40), nn.Linear(40, 20), nn.Linear(20, 10))

    total_params = 0
    for i, layer in enumerate(model):
        if isinstance(layer, nn.Linear):
            params = layer.weight.numel() + layer.bias.numel()
            total_params += params
            print(f"第{i+1}层: {layer.in_features} → {layer.out_features}")
            print(f"  权重参数: {layer.weight.numel()}")
            print(f"  偏置参数: {layer.bias.numel()}")
            print(f"  总参数: {params}")
            print()

    print(f"总参数数量: {total_params}")


if __name__ == "__main__":
    demonstrate_feature_compression()
    explain_why_linear()
    show_parameter_details()
