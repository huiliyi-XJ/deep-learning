import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

print("=== 自定义数据集处理示例 ===\n")

# 方法1：直接构造数据
print("方法1：直接构造NumPy数组")
print("-" * 40)

# 构造学生数据：学习时间、睡眠时间、运动时间、是否通过考试
np.random.seed(42)  # 固定随机种子，确保结果可复现

# 生成100个学生的数据
n_samples = 100

# 特征：学习时间(小时)、睡眠时间(小时)、运动时间(小时)
study_time = np.random.uniform(1, 8, n_samples)  # 1-8小时
sleep_time = np.random.uniform(6, 10, n_samples)  # 6-10小时
exercise_time = np.random.uniform(0, 3, n_samples)  # 0-3小时

# 组合特征矩阵
X = np.column_stack([study_time, sleep_time, exercise_time])

# 生成标签：基于特征计算通过概率，然后二值化
# 通过概率 = 学习时间*0.4 + 睡眠时间*0.3 + 运动时间*0.2 + 随机噪声
pass_probability = (
    study_time * 0.4
    + sleep_time * 0.3
    + exercise_time * 0.2
    + np.random.normal(0, 0.1, n_samples)
)
y = (pass_probability > np.median(pass_probability)).astype(np.float32)

print(f"特征矩阵形状: {X.shape}")  # (100, 3)
print(f"标签向量形状: {y.shape}")  # (100,)
print(f"通过率: {y.mean():.2%}")

# 转换为PyTorch张量
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).float().view(-1, 1)

print(f"PyTorch特征张量形状: {X_tensor.shape}")  # (100, 3)
print(f"PyTorch标签张量形状: {y_tensor.shape}")  # (100, 1)

# 创建数据集
dataset = TensorDataset(X_tensor, y_tensor)
print(f"数据集大小: {len(dataset)}")

# 创建数据加载器
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# 查看第一个批次
for batch_x, batch_y in dataloader:
    print(f"\n第一个批次:")
    print(f"  特征形状: {batch_x.shape}")  # (10, 3)
    print(f"  标签形状: {batch_y.shape}")  # (10, 1)
    print(f"  特征值示例: {batch_x[0]}")  # 第一个样本的特征
    print(f"  标签值示例: {batch_y[0]}")  # 第一个样本的标签
    break

print("\n" + "=" * 50)

# 方法2：从CSV文件读取（模拟）
print("方法2：从CSV文件读取")
print("-" * 40)

# 创建示例CSV数据
csv_data = {
    "study_hours": study_time,
    "sleep_hours": sleep_time,
    "exercise_hours": exercise_time,
    "passed": y,
}

# 保存为CSV文件
df = pd.DataFrame(csv_data)
df.to_csv("student_data.csv", index=False)
print("已创建 student_data.csv 文件")

# 读取CSV文件
df_loaded = pd.read_csv("student_data.csv")

# 分离特征和标签
feature_columns = ["study_hours", "sleep_hours", "exercise_hours"]
X_csv = df_loaded[feature_columns].values
y_csv = df_loaded["passed"].values

print(f"从CSV读取的特征形状: {X_csv.shape}")
print(f"从CSV读取的标签形状: {y_csv.shape}")

# 转换为PyTorch张量
X_csv_tensor = torch.from_numpy(X_csv).float()
y_csv_tensor = torch.from_numpy(y_csv).float().view(-1, 1)

print(f"CSV数据转换为张量后:")
print(f"  特征张量形状: {X_csv_tensor.shape}")
print(f"  标签张量形状: {y_csv_tensor.shape}")

print("\n" + "=" * 50)

# 方法3：数据预处理示例
print("方法3：数据预处理")
print("-" * 40)

# 标准化特征（重要！）
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_csv)

print("标准化前后对比:")
print(f"原始特征均值: {X_csv.mean(axis=0)}")
print(f"标准化后均值: {X_scaled.mean(axis=0)}")
print(f"标准化后标准差: {X_scaled.std(axis=0)}")

# 转换为PyTorch张量
X_scaled_tensor = torch.from_numpy(X_scaled).float()
y_csv_tensor = torch.from_numpy(y_csv).float().view(-1, 1)

print(f"\n最终处理后的数据:")
print(f"特征张量形状: {X_scaled_tensor.shape}")
print(f"标签张量形状: {y_csv_tensor.shape}")

# 创建最终数据集
final_dataset = TensorDataset(X_scaled_tensor, y_csv_tensor)
final_dataloader = DataLoader(final_dataset, batch_size=16, shuffle=True)

print(f"最终数据集大小: {len(final_dataset)}")
print(f"批次数: {len(final_dataloader)}")

print("\n=== 数据准备完成，可以开始训练模型！ ===")
