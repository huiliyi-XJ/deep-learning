import numpy as np
import matplotlib.pyplot as plt

# 实现定义好数据集的x值和y值
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


# 定义预测函数，因为是要确定损失值随着w的变化情况
# 后续会给定w的值，所以参数只有x
def forward(x):
    return x * w


# 定义损失函数的计算方式，仅针对一个样本而言
def loss(x, y):
    y_pred = forward(x)
    return (y - y_pred) * (y - y_pred)


# 程序的目的是为了比较mse和w之间的关系，所以要把其二者的结果对应放入列表中
w_list = []
mse_list = []

# 穷举法遍历所有可能的w值，并计算对应的mse值
for w in np.arange(0.0, 4.1, 0.1):
    print("w = ", w)
    l_sum = 0  # 存放所有样本的损失函数值
    for x_val, y_val in zip(x_data, y_data):
        y_pred_val = forward(x_val)
        loss_val = loss(x_val, y_val)
        l_sum += loss_val
        print("\t", x_val, y_val, y_pred_val, loss_val)
    print("MSE=", l_sum / 3)
    w_list.append(w)
    mse_list.append(l_sum / 3)

plt.plot(w_list, mse_list)  # 绘制mse和w之间的关系
plt.ylabel("Loss")
plt.xlabel("w")
plt.show()
