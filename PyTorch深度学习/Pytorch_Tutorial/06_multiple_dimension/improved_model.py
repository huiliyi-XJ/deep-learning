import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader, random_split

print("=== 改进版糖尿病预测模型 ===\n")

# 1. 数据加载和预处理
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

print("原始数据信息:")
print(f"特征形状: {X.shape}")
print(f"目标值范围: {y.min():.1f} - {y.max():.1f}")
print(f"目标值均值: {y.mean():.1f}")
print(f"目标值标准差: {y.std():.1f}")

# 2. 特征标准化（重要！）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\n标准化后特征统计:")
print(f"特征均值: {X_scaled.mean(axis=0)}")
print(f"特征标准差: {X_scaled.std(axis=0)}")

# 3. 标签处理 - 使用更合理的阈值
# 方法1：使用分位数而不是中位数
threshold = np.percentile(y, 70)  # 70%分位数作为阈值
y_binary = (y > threshold).astype(np.float32)

print(f"\n标签处理:")
print(f"使用阈值: {threshold:.1f}")
print(f"正样本比例: {y_binary.mean():.2%}")

# 4. 转换为PyTorch张量
X_tensor = torch.from_numpy(X_scaled).float()
y_tensor = torch.from_numpy(y_binary).float().view(-1, 1)

# 5. 数据集划分
dataset = TensorDataset(X_tensor, y_tensor)
total_samples = len(dataset)
train_size = int(0.7 * total_samples)
val_size = int(0.2 * total_samples)
test_size = total_samples - train_size - val_size

generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size], generator=generator
)


# 6. 改进的模型结构
class ImprovedModel(torch.nn.Module):
    def __init__(self, input_size=10, hidden_sizes=[16, 8, 4], dropout_rate=0.3):
        super(ImprovedModel, self).__init__()

        layers = []
        prev_size = input_size

        # 构建隐藏层
        for hidden_size in hidden_sizes:
            layers.extend(
                [
                    torch.nn.Linear(prev_size, hidden_size),
                    torch.nn.BatchNorm1d(hidden_size),  # 批归一化
                    torch.nn.ReLU(),  # ReLU激活函数
                    torch.nn.Dropout(dropout_rate),  # Dropout防止过拟合
                ]
            )
            prev_size = hidden_size

        # 输出层
        layers.append(torch.nn.Linear(prev_size, 1))
        layers.append(torch.nn.Sigmoid())

        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# 7. 创建改进的模型
model = ImprovedModel(input_size=10, hidden_sizes=[16, 8, 4], dropout_rate=0.3)

# 8. 损失函数和优化器
criterion = torch.nn.BCELoss(reduction="mean")  # 使用mean而不是sum
optimizer = torch.optim.Adam(
    model.parameters(), lr=0.001, weight_decay=1e-4
)  # 添加L2正则化
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5
)

# 9. 数据加载器
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 10. 训练循环
best_val_loss = float("inf")
patience = 15
patience_counter = 0
train_losses = []
val_losses = []

print(f"\n开始训练...")
print(f"训练集大小: {len(train_dataset)}")
print(f"验证集大小: {len(val_dataset)}")
print(f"测试集大小: {len(test_dataset)}")

for epoch in range(200):  # 增加训练轮数
    # 训练阶段
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        y_pred = model(batch_x)
        loss = criterion(y_pred, batch_y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        predicted = (y_pred > 0.5).float()
        train_correct += (predicted == batch_y).sum().item()
        train_total += batch_y.size(0)

    # 验证阶段
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            y_pred = model(batch_x)
            loss = criterion(y_pred, batch_y)
            val_loss += loss.item()

            predicted = (y_pred > 0.5).float()
            val_correct += (predicted == batch_y).sum().item()
            val_total += batch_y.size(0)

    # 计算平均损失和准确率
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    train_acc = 100 * train_correct / train_total
    val_acc = 100 * val_correct / val_total

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    # 学习率调度
    scheduler.step(avg_val_loss)

    # 早停机制
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_improved.pt")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"早停在第 {epoch+1} 轮")
            break

    # 打印进度
    if (epoch + 1) % 20 == 0:
        print(
            f"Epoch {epoch+1:3d}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.2f}%, "
            f"Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.2f}%"
        )

# 11. 测试最佳模型
model.load_state_dict(torch.load("best_improved.pt"))
model.eval()

test_correct = 0
test_total = 0
test_predictions = []
test_true_labels = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        y_pred = model(batch_x)
        predicted = (y_pred > 0.5).float()
        test_correct += (predicted == batch_y).sum().item()
        test_total += batch_y.size(0)

        test_predictions.extend(y_pred.cpu().numpy().flatten())
        test_true_labels.extend(batch_y.cpu().numpy().flatten())

test_accuracy = 100 * test_correct / test_total

print(f"\n=== 最终结果 ===")
print(f"测试准确率: {test_accuracy:.2f}%")
print(f"测试样本数: {test_total}")

# 12. 详细分析
from sklearn.metrics import classification_report, confusion_matrix

test_predictions_binary = (np.array(test_predictions) > 0.5).astype(int)
test_true_labels = np.array(test_true_labels).astype(int)

print(f"\n=== 分类报告 ===")
print(
    classification_report(
        test_true_labels,
        test_predictions_binary,
        target_names=["糖尿病轻微", "糖尿病严重"],
    )
)

print(f"\n=== 混淆矩阵 ===")
cm = confusion_matrix(test_true_labels, test_predictions_binary)
print(cm)

# 计算精确率、召回率、F1分数
tn, fp, fn, tp = cm.ravel()
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n=== 详细指标 ===")
print(f"精确率 (Precision): {precision:.3f}")
print(f"召回率 (Recall): {recall:.3f}")
print(f"F1分数: {f1:.3f}")

print(f"\n=== 改进总结 ===")
print("1. 特征标准化")
print("2. 更合理的标签阈值")
print("3. 改进的模型结构（BatchNorm + ReLU + Dropout）")
print("4. 更好的优化器设置（Adam + 学习率调度 + L2正则化）")
print("5. 更详细的评估指标")
