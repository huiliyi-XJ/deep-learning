import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# 实现定义好数据集的x值和y值
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


# 定义预测函数，因为是要确定损失值随着w的变化情况
# 后续会给定w,b的值，所以参数只有x
def forward(x):
    return x * w + b


# 定义损失函数的计算方式，仅针对一个样本而言
def loss(x, y):
    y_pred = forward(x)
    return (y - y_pred) * (y - y_pred)


# 程序的目的是为了比较mse和w之间的关系，所以要把其二者的结果对应放入列表中
w_list = np.arange(0.0, 4.1, 0.1)
b_list = np.arange(-2.0, 2.1, 0.1)
mse_list = []
ww, bb = np.meshgrid(w_list, b_list)
# 根据老师说的【穷举法】思想，在一个确定的范围找到mse和w、b之间的关系
for b in b_list:
    for w in w_list:
        print("w ={0},b = {1}".format(w, b))
        l_sum = 0  # 存放所有样本的损失函数值
        for x_val, y_val in zip(x_data, y_data):
            y_pred_val = forward(x_val)
            loss_val = loss(x_val, y_val)
            l_sum += loss_val
            print("\t", x_val, y_val, y_pred_val, loss_val)
        print("MSE=", l_sum / 3)
        mse_list.append(l_sum / 3)
mse = np.array(mse_list).reshape(w_list.shape[0], b_list.shape[0])

# h = plt.contourf(w_list, b_list, mse)  # 绘制二维等高线图
fig = plt.figure()
ax = Axes3D(fig)
plt.ylabel("b")
plt.xlabel("w")
ax.plot_surface(ww, bb, mse, rstride=1, cstride=1, cmap=cm.viridis)

plt.show()
