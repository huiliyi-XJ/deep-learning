import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


# 模拟一个简单的CNN数据流
def visualize_data_flow():
    # 模拟输入数据
    batch_size = 2  # 为了可视化，使用小批次
    input_data = torch.randn(batch_size, 1, 28, 28)

    print("=== CNN数据流动过程可视化 ===\n")

    # 第一层卷积+池化
    conv1 = torch.nn.Conv2d(1, 10, kernel_size=2)
    pooling = torch.nn.MaxPool2d(2)

    x = input_data
    print(f"1. 输入图像: {x.shape}")

    x = conv1(x)
    print(f"2. 第一层卷积后: {x.shape}")

    x = pooling(x)
    print(f"3. 第一层池化后: {x.shape}")

    # 第二层卷积+池化
    conv2 = torch.nn.Conv2d(10, 20, kernel_size=2)

    x = conv2(x)
    print(f"4. 第二层卷积后: {x.shape}")

    x = pooling(x)
    print(f"5. 第二层池化后: {x.shape}")

    # 第三层卷积+池化
    conv3 = torch.nn.Conv2d(20, 20, kernel_size=2)

    x = conv3(x)
    print(f"6. 第三层卷积后: {x.shape}")

    x = pooling(x)
    print(f"7. 第三层池化后: {x.shape}")

    # 展平操作
    print(f"\n=== 关键步骤：展平操作 ===")
    print(f"展平前: {x.shape}")
    print(
        f"展平计算: {x.shape[0]} × {x.shape[1]} × {x.shape[2]} × {x.shape[3]} = {x.shape[0]} × {x.shape[1] * x.shape[2] * x.shape[3]}"
    )

    x = x.view(batch_size, -1)
    print(f"展平后: {x.shape}")

    # 全连接层
    linear1 = torch.nn.Linear(80, 40)
    linear2 = torch.nn.Linear(40, 20)
    linear3 = torch.nn.Linear(20, 10)

    x = linear1(x)
    print(f"8. 第一层全连接后: {x.shape}")

    x = linear2(x)
    print(f"9. 第二层全连接后: {x.shape}")

    x = linear3(x)
    print(f"10. 第三层全连接后: {x.shape}")

    print(f"\n=== 最终输出 ===")
    print(f"每个样本输出10个数字，对应0-9的概率分布")
    print(f"输出形状: {x.shape}")


def explain_flatten():
    print("\n=== 展平操作详细解释 ===")
    print("x.view(batch_size, -1) 的含义：")
    print("1. batch_size: 保持批次大小不变")
    print("2. -1: 自动计算剩余维度")
    print("3. 将 (batch, channels, height, width) 转换为 (batch, features)")

    # 示例
    example = torch.randn(3, 20, 2, 2)
    print(f"\n示例张量形状: {example.shape}")
    print(f"展平后形状: {example.view(3, -1).shape}")
    print(f"计算: 3 × 20 × 2 × 2 = 3 × 80")


if __name__ == "__main__":
    visualize_data_flow()
    explain_flatten()
