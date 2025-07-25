# 训练数据：输入特征
x_data = [1.0, 2.0, 3.0]
# 训练数据：目标输出（标签）
y_data = [2.0, 4.0, 6.0]

# 初始化权重参数，这里假设真实的权重是2.0
w = 1.02


# 定义线性模型：y = w * x
def forward(x):
    return x * w


# 计算均方误差损失函数
def cost(xs, ys):
    cost = 0  # 初始化损失值
    for x, y in zip(xs, ys):  # 遍历所有训练样本
        y_pred = forward(x)  # 使用当前权重进行预测
        cost += (y_pred - y) ** 2  # 累加每个样本的平方误差
    return cost / len(xs)  # 返回平均损失（均方误差）


# 计算损失函数对权重w的梯度
def gradient(xs, ys):
    grad = 0  # 初始化梯度值
    for x, y in zip(xs, ys):  # 遍历所有训练样本
        # 计算梯度：∂L/∂w = 2 * x * (w*x - y)
        # 其中 L = (w*x - y)²，对w求导得到 2*(w*x - y)*x
        grad += 2 * x * (x * w - y)
    return grad / len(xs)  # 返回平均梯度


# 在训练前预测x=4时的输出值
print("Predict (before training)", 4, forward(4))

# 开始训练循环，进行100轮迭代
for epoch in range(1000):
    # 设置学习率，控制每次权重更新的步长
    lr = 0.01
    # 计算当前轮次的损失值
    cost_val = cost(x_data, y_data)
    # 计算当前轮次的梯度值
    grad_val = gradient(x_data, y_data)
    # 使用梯度下降更新权重：w = w - lr * gradient
    w -= lr * grad_val
    # 打印当前轮次的信息：轮次、权重值、损失值
    print("Epoch:", epoch, "w=", w, "loss=", cost_val)

# 在训练后预测x=4时的输出值，验证训练效果
print("Predict (after training)", 4, forward(4))
