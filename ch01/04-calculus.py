import numpy as np
from d2l import torch as d2l
import os

# 为了解决可能的库冲突问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 定义一个函数 f(x)
def f(x):
    return 3 * x ** 2 - 4 * x  # 这是一个简单的二次函数 f(x) = 3x^2 - 4x

# 定义数值微分函数，用于近似求导
def numerical_lim(f, x, h):
    return (f(x + h) - f(x)) / h  # 使用有限差分法计算 f(x) 在 x 处的导数

# 初始化步长 h
h = 0.1
# 通过循环逐渐减小 h 来观察数值导数的变化
for i in range(5):
    print(f'h={h:.5f}, numerical limit={numerical_lim(f, 1, h):.5f}')
    h *= 0.1  # 每次循环将 h 乘以 0.1，使其逐渐减小

# 生成从 0 到 3 的 x 值，步长为 0.1
x = np.arange(0, 3, 0.1)
# 绘制函数 f(x) 及其在 x=1 处的切线，切线方程为 y = 2x - 3
d2l.plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
# 显示图像
d2l.plt.show()

# 生成从 0.5 到 3 的 x 值，步长为 0.2
x = np.arange(0.5, 3, 0.2)
# 绘制另一个函数 f(x) = x^3 - 1/x 及其在 x=1 处的切线，切线方程为 y = 4x - 4
d2l.plot(x, [x ** 3 - 1 / x, 4 * x - 4], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
# 显示图像
d2l.plt.show()
