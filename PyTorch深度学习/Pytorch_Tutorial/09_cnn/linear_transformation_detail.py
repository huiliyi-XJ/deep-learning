import torch
import torch.nn as nn
import numpy as np


def explain_linear_transformation():
    print("=== 80维到40维的线性变换详解 ===\n")

    # 创建输入数据
    input_features = torch.randn(1, 80)
    print(f"输入特征 (80维): {input_features.shape}")
    print(f"前10个特征值: {input_features[0, :10]}")

    # 创建线性层
    linear_layer = nn.Linear(80, 40)

    print(f"\n=== 权重矩阵形状 ===")
    print(f"权重矩阵 W: {linear_layer.weight.shape}")  # (40, 80)
    print(f"偏置向量 b: {linear_layer.bias.shape}")  # (40,)

    # 手动计算第一个输出
    print(f"\n=== 手动计算第一个输出 z1 ===")
    w1 = linear_layer.weight[0]  # 第一行的权重 (80,)
    b1 = linear_layer.bias[0]  # 第一个偏置

    print(f"第一个输出的权重: {w1.shape}")
    print(f"第一个输出的偏置: {b1.item():.4f}")

    # 计算 z1 = w11*x1 + w12*x2 + ... + w1,80*x80 + b1
    z1_manual = torch.dot(w1, input_features[0]) + b1
    print(f"手动计算的 z1: {z1_manual.item():.4f}")

    # 使用线性层计算
    output = linear_layer(input_features)
    z1_auto = output[0, 0]
    print(f"自动计算的 z1: {z1_auto.item():.4f}")
    print(f"两者是否相等: {torch.abs(z1_manual - z1_auto) < 1e-6}")

    return input_features, linear_layer, output


def show_weight_analysis():
    print(f"\n=== 权重分析 ===")

    # 创建线性层
    linear_layer = nn.Linear(80, 40)
    weights = linear_layer.weight  # (40, 80)

    print(f"权重矩阵形状: {weights.shape}")
    print(f"权重值范围: [{weights.min():.4f}, {weights.max():.4f}]")
    print(f"权重均值: {weights.mean():.4f}")
    print(f"权重标准差: {weights.std():.4f}")

    # 分析每个输出特征使用了多少输入特征
    print(f"\n=== 每个输出特征的特征使用情况 ===")

    # 计算每个输出特征的非零权重数量
    nonzero_counts = (weights != 0).sum(dim=1)
    print(f"每个输出特征使用的输入特征数: {nonzero_counts}")
    print(f"平均使用特征数: {nonzero_counts.float().mean():.1f}")

    # 显示前5个输出特征的权重分布
    print(f"\n=== 前5个输出特征的权重分布 ===")
    for i in range(5):
        w_i = weights[i]
        print(f"输出特征 {i+1}:")
        print(f"  权重范围: [{w_i.min():.4f}, {w_i.max():.4f}]")
        print(f"  非零权重数: {(w_i != 0).sum().item()}")
        print(f"  最大权重索引: {w_i.argmax().item()}")
        print(f"  最小权重索引: {w_i.argmin().item()}")


def demonstrate_feature_combination():
    print(f"\n=== 特征组合演示 ===")

    # 创建简化的例子
    input_simple = torch.tensor([[1.0, 2.0, 3.0, 4.0]])  # 4维输入
    linear_simple = nn.Linear(4, 2)  # 4维到2维

    print(f"简化输入: {input_simple}")
    print(f"输入形状: {input_simple.shape}")

    # 显示权重
    print(f"权重矩阵:\n{linear_simple.weight}")
    print(f"偏置向量: {linear_simple.bias}")

    # 计算输出
    output_simple = linear_simple(input_simple)
    print(f"输出: {output_simple}")
    print(f"输出形状: {output_simple.shape}")

    # 手动计算
    w = linear_simple.weight
    b = linear_simple.bias
    x = input_simple[0]

    z1 = w[0, 0] * x[0] + w[0, 1] * x[1] + w[0, 2] * x[2] + w[0, 3] * x[3] + b[0]
    z2 = w[1, 0] * x[0] + w[1, 1] * x[1] + w[1, 2] * x[2] + w[1, 3] * x[3] + b[1]

    print(f"手动计算:")
    print(
        f"  z1 = {w[0, 0]:.4f}*{x[0]:.1f} + {w[0, 1]:.4f}*{x[1]:.1f} + {w[0, 2]:.4f}*{x[2]:.1f} + {w[0, 3]:.4f}*{x[3]:.1f} + {b[0]:.4f} = {z1:.4f}"
    )
    print(
        f"  z2 = {w[1, 0]:.4f}*{x[0]:.1f} + {w[1, 1]:.4f}*{x[1]:.1f} + {w[1, 2]:.4f}*{x[2]:.1f} + {w[1, 3]:.4f}*{x[3]:.1f} + {b[1]:.4f} = {z2:.4f}"
    )


def explain_why_not_simple_combination():
    print(f"\n=== 为什么不是简单的两两组合？ ===")
    print("1. 信息损失:")
    print("   - 简单组合会丢失很多信息")
    print("   - 每个输出特征只使用2个输入特征")
    print("   - 其他78个特征的信息被忽略")

    print("\n2. 表达能力:")
    print("   - 全连接层可以学习任意线性组合")
    print("   - 每个输出特征可以使用所有输入特征")
    print("   - 更强的特征表达能力")

    print("\n3. 学习能力:")
    print("   - 网络可以自动学习最优的特征组合")
    print("   - 权重通过反向传播自动调整")
    print("   - 适应不同的数据分布")


if __name__ == "__main__":
    input_features, linear_layer, output = explain_linear_transformation()
    show_weight_analysis()
    demonstrate_feature_combination()
    explain_why_not_simple_combination()
