import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from torch.utils.data import TensorDataset, DataLoader, random_split

print("=== 高级版糖尿病预测模型 ===\n")

# 1. 数据加载
diabetes = load_diabetes()
X = diabete
y = diabetes.target

print("原始数据信息:")
print(f"特征形状: {X.shape}")
print(f"特征名称: {diabetes.feature_names}")

# 2. 特征工程 - 添加交互特征
print("\n=== 特征工程 ===")

# 原始特征
X_original = X.copy()

# 添加特征交互
X_interactions = np.column_stack(
    [
        X_original,
        X_original[:, 0] * X_original[:, 1],  # 年龄 * 性别
        X_original[:, 2] * X_original[:, 3],  # BMI * 血压
        X_original[:, 4] * X_original[:, 5],  # 总胆固醇 * LDL
        X_original[:, 6] * X_original[:, 7],  # HDL * TCH/HDL
        X_original[:, 8] * X_original[:, 9],  # 甘油三酯 * 血糖
        X_original[:, 2] ** 2,  # BMI的平方
        X_original[:, 4] ** 2,  # 总胆固醇的平方
        X_original[:, 9] ** 2,  # 血糖的平方
    ]
)

print(f"特征工程后形状: {X_interactions.shape}")

# 3. 特征选择 - 选择最重要的特征
selector = SelectKBest(score_func=f_classif, k=15)
X_selected = selector.fit_transform(X_interactions, y)

# 获取选中的特征索引
selected_features = selector.get_support()
print(f"选中的特征数量: {X_selected.shape[1]}")

# 4. 数据标准化 - 使用RobustScaler处理异常值
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_selected)

print(f"\n标准化后特征统计:")
print(f"特征均值: {X_scaled.mean(axis=0)[:5]}...")  # 显示前5个
print(f"特征标准差: {X_scaled.std(axis=0)[:5]}...")

# 5. 标签处理 - 尝试多个阈值
thresholds = [np.percentile(y, 60), np.percentile(y, 70), np.percentile(y, 80)]
best_threshold = None
best_balance = float("inf")

for threshold in thresholds:
    y_binary = (y > threshold).astype(np.float32)
    balance = abs(y_binary.mean() - 0.5)  # 越接近0.5越平衡
    if balance < best_balance:
        best_balance = balance
        best_threshold = threshold

y_binary = (y > best_threshold).astype(np.float32)
print(f"\n标签处理:")
print(f"使用阈值: {best_threshold:.1f}")
print(f"正样本比例: {y_binary.mean():.2%}")

# 6. 转换为PyTorch张量
X_tensor = torch.from_numpy(X_scaled).float()
y_tensor = torch.from_numpy(y_binary).float().view(-1, 1)

# 7. 数据集划分
dataset = TensorDataset(X_tensor, y_tensor)
total_samples = len(dataset)
train_size = int(0.7 * total_samples)
val_size = int(0.2 * total_samples)
test_size = total_samples - train_size - val_size

generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size], generator=generator
)


# 8. 高级模型结构
class AdvancedModel(torch.nn.Module):
    def __init__(self, input_size=15, dropout_rate=0.4):
        super(AdvancedModel, self).__init__()

        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Linear(input_size, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(32, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(64, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(32, 16),
            torch.nn.BatchNorm1d(16),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(16, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output


# 9. 创建模型
model = AdvancedModel(input_size=X_scaled.shape[1], dropout_rate=0.4)

# 10. 损失函数和优化器
criterion = torch.nn.BCELoss(reduction="mean")
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2
)

# 11. 数据加载器
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# 12. 训练循环
best_val_loss = float("inf")
patience = 20
patience_counter = 0
best_epoch = 0

print(f"\n开始训练...")
print(f"输入特征数: {X_scaled.shape[1]}")
print(f"训练集大小: {len(train_dataset)}")
print(f"验证集大小: {len(val_dataset)}")
print(f"测试集大小: {len(test_dataset)}")

for epoch in range(300):
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

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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

    # 计算指标
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    train_acc = 100 * train_correct / train_total
    val_acc = 100 * val_correct / val_total

    # 学习率调度
    scheduler.step()

    # 早停机制
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_advanced.pt")
        patience_counter = 0
        best_epoch = epoch
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"早停在第 {epoch+1} 轮 (最佳轮次: {best_epoch+1})")
            break

    # 打印进度
    if (epoch + 1) % 30 == 0:
        print(
            f"Epoch {epoch+1:3d}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.2f}%, "
            f"Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.2f}%"
        )

# 13. 测试最佳模型
model.load_state_dict(torch.load("best_advanced.pt"))
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

# 14. 详细分析
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

test_predictions_binary = (np.array(test_predictions) > 0.5).astype(int)
test_true_labels = np.array(test_true_labels).astype(int)

# ROC AUC分数
roc_auc = roc_auc_score(test_true_labels, test_predictions)

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

# 计算详细指标
tn, fp, fn, tp = cm.ravel()
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

print(f"\n=== 详细指标 ===")
print(f"精确率 (Precision): {precision:.3f}")
print(f"召回率 (Recall): {recall:.3f}")
print(f"特异度 (Specificity): {specificity:.3f}")
print(f"F1分数: {f1:.3f}")
print(f"ROC AUC: {roc_auc:.3f}")

print(f"\n=== 改进总结 ===")
print("1. 特征工程：添加交互特征和多项式特征")
print("2. 特征选择：选择最重要的15个特征")
print("3. 鲁棒标准化：处理异常值")
print("4. 平衡标签：优化正负样本比例")
print("5. 深度网络：更复杂的网络结构")
print("6. 高级优化：AdamW + 余弦退火学习率")
print("7. 梯度裁剪：防止梯度爆炸")
print("8. 详细评估：ROC AUC等更多指标")
