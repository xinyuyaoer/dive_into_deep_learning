import torch

# 1. 自动梯度计算
print('1.自动梯度计算')
# 创建一个向量 x，包含4个元素，并启用梯度计算
x = torch.arange(4.0, requires_grad=True)  
# 1.将梯度附加到想要对其计算偏导数的变量
# requires_grad=True: 表示我们希望对 x 计算梯度。x 的每个元素都将被标记为需要计算梯度。
print('x:', x)
# x.grad 此时还没有梯度，因为我们还没有进行反向传播
print('x.grad:', x.grad)


# 计算目标值 y，y = 2 * (x与x的点积)
y = 2 * torch.dot(x, x)  # 2.记录目标值的计算
print('y:', y)

# 执行反向传播，计算 y 对 x 的偏导数并存储在 x.grad 中
y.backward()  # 3.执行它的反向传播函数
print('x.grad:', x.grad)  # 4.访问得到的梯度

#* 验证计算结果，y = 2 * (x与x的点积) = 2 * (x1^2 + x2^2 + ... + x4^2)
#* 对 x 求导：dy/dx = 4 * x，因此验证是否等于 4 * x
print('x.grad == 4*x:', x.grad == 4 * x)

# 计算另一个函数
# 在继续计算前将梯度清零，避免累加前一次的梯度
# （在默认情况下，PyTorch会累积梯度，我们需要清除之前的值）
x.grad.zero_()
y = x.sum()  # y = x1 + x2 + x3 + x4
print('y:', y)

# 对 y 进行反向传播，计算 y 对 x 的梯度
y.backward()
print('x.grad:', x.grad)  # 对于 sum 函数，所有分量的梯度都为 1

# 非标量变量的反向传播
x.grad.zero_()  # 再次将梯度清零
print('x:', x)
y = x * x  # y 是一个与 x 形状相同的张量，每个元素为 x 的平方
y.sum().backward()  # 对 y.sum() 进行反向传播
print('x.grad:', x.grad)  # 由于 y = x^2，梯度应为 2x

#* 将某些计算移动到记录的计算图之外
#* detach() 方法用于将某些张量从计算图中分离出来，通常用于停止梯度流动，或者手动控制哪些部分需要参与自动微分。
# 清零之前计算的梯度
x.grad.zero_()
# 计算 y = x * x
y = x * x
# detach() 方法将 y 从计算图中分离出来，产生一个新的张量 u
# u 与 y 的值相同，但 u 不会记录在计算图中，因此对 u 的操作不会影响原始计算图
u = y.detach()
# 计算 z = u * x，但由于 u 已经从计算图中分离，因此 z 不会影响 y 的梯度计算
z = u * x
# 对 z 的和进行反向传播，计算 z 对 x 的梯度
z.sum().backward()
# 检查 x.grad 是否等于 u，这里不会相等，因为 u 已经从计算图中分离，梯度计算并不考虑 u 的生成过程
x.grad == u



# 即使构建函数的计算图需要通过Python控制流（例如，条件、循环或任意函数调用，我们仍然可以计算得到的变量的梯度）
# 定义一个复杂的函数 f(a)
def f(a):
    b = a * 2
    print(b.norm())  # 计算 b 的 L2 范数（向量的模）
    while b.norm() < 1000:  # 当 b 的 L2 范数小于 1000 时，继续扩大 b
        b = b * 2
    if b.sum() > 0:  # 判断 b 的元素和是否大于 0
        c = b
    else:
        c = 100 * b
    return c  # 返回 c

# 2. Python控制流的梯度计算
print('2.Python控制流的梯度计算')
# 初始化一个标量变量 a
a = torch.tensor(2.0)  # 初始化变量
a.requires_grad_(True)  # 1.将梯度附给想要对其求偏导数的变量
print('a:', a)

# 将 a 传入函数 f 计算出结果 d
d = f(a)  # 2.记录目标函数
print('d:', d)

# 对 d 进行反向传播，计算 d 对 a 的梯度
d.backward()  # 3.执行目标函数的反向传播函数
print('a.grad:', a.grad)  # 4.获取梯度

# 由于函数 f 中包含了 Python 的控制流（如循环、条件判断），PyTorch 依然能够自动计算梯度
# 这是因为 PyTorch 记录了所有张量操作的计算图，即使这些操作涉及到 Python 的控制流
