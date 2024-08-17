import torch

print('1.标量与变量') # 标量由只有一个元素的张量表示
x = torch.tensor([3.0])
y = torch.tensor([2.0])
print(x + y, x * y, x / y, x ** y)

x = torch.arange(4) # 可以将向量视为标量值组成的列表
print('2.向量')
print('x:', x)
print('x[3]:', x[3])  # 通过张量的索引来访问任一元素
print('张量的形状:', x.shape)  # 张量的形状
print('张量的长度:', len(x))  # 张量的长度
z = torch.arange(24).reshape(2, 3, 4)
print('三维张量的长度:', len(z))

print('3.矩阵')
A = torch.arange(20).reshape(5, 4)
print('A:', A)
print('A.shape:', A.shape)
print('A.shape[-1]:', A.shape[-1]) #A.shape[-1] 表示访问 A 形状中的最后一个维度的大小
print('A.T:', A.T)  # 矩阵的转置

print('4.矩阵的计算')
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()  # 通过分配新内存，将A的一个副本分配给B
print('A:', A)
print('B:', B)
print('A + B:', A + B)  # 矩阵相加
print('A * B:', A * B)  # 矩阵相乘

a = 2
X = torch.arange(24).reshape(2, 3, 4) #* 这里的 (2, 3, 4) 表示张量有 2 个矩阵（或 "batch"），每个矩阵有 3 行和 4 列。
print('X:', X)
print('a + X:', a + X)  # 矩阵的值加上标量
print('a * X:', a * X)
print((a * X).shape)

print('5.矩阵的sum运算')
print('A:', A)
print('A.shape:', A.shape)
print('A.sum():', A.sum()) #A.sum(): 对张量 A 中的所有元素进行求和
print('A.sum(axis=0):', A.sum(axis=0))  
# A.sum(axis=0): 沿着第 0 轴（即沿着行方向）对 A 进行求和，这会生成一个大小为 (4,) 的向量，其中每个元素是 A 中对应列的和。
# 结果: [0+4+8+12+16, 1+5+9+13+17, 2+6+10+14+18, 3+7+11+15+19]
print('A.sum(axis=1):', A.sum(axis=1))  
# A.sum(axis=1): 沿着第 1 轴（即沿着列方向）对 A 进行求和，这会生成一个大小为 (5,) 的向量，其中每个元素是 A 中对应行的和。
# 结果: [0+1+2+3, 4+5+6+7, 8+9+10+11, 12+13+14+15, 16+17+18+19]
print('A.sum(axis=1, keepdims=True)', A.sum(axis=1, keepdims=True))  
# A.sum(axis=1, keepdims=True): 沿第 1 轴进行求和，同时使用 keepdims=True 保持原有的维度数量。这意味着结果将是一个大小为 (5, 1) 的张量，而不是 (5,)。
# 通过广播机制，A.sum(axis=1, keepdims=True) 保持了原始张量的形状，使得可以与其他张量（如原始张量 A）进行元素级别的操作。广播在操作多维张量时非常有用，尤其是在需要保持或扩展维度以进行计算时。
print('A.sum(axis=[0, 1]):', A.sum(axis=[0, 1]))  
# Same as `A.sum()`
# A.sum(axis=[0, 1]): 沿第 0 轴和第 1 轴进行求和，相当于对整个张量 A 中的所有元素进行求和。这和直接调用 A.sum() 的结果是一样的。
print('A.mean():', A.mean())
print('A.sum() / A.numel():', A.sum() / A.numel())
print('A.cumsum(axis=0):', A.cumsum(axis=0)) 
# axis=0 表示沿着第 0 轴（行方向）进行累积总和。这意味着，每个元素都是该列当前行和前面所有行的累积和。
""" 假设 A 是如下的张量：
tensor([[ 0.,  1.,  2.,  3.],
        [ 4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11.],
        [12., 13., 14., 15.],
        [16., 17., 18., 19.]])
计算 A.cumsum(axis=0) 的结果为：
tensor([[ 0.,  1.,  2.,  3.],
        [ 4.,  6.,  8., 10.],
        [12., 15., 18., 21.],
        [24., 28., 32., 36.],
        [40., 45., 50., 55.]])
第一行保持不变，因为它没有前面的行。
第二行中的每个元素是第一行和第二行相应元素的累积和，例如：4 = 0 + 4, 6 = 1 + 5 等。
第三行中的每个元素是第一行、第二行和第三行相应元素的累积和，依此类推。
 """
print('6.向量-向量相乘（点积）')
x = torch.arange(4, dtype=torch.float32)
y = torch.ones(4, dtype=torch.float32)
print('x:', x)
print('y:', y)
print('向量-向量点积:', torch.dot(x, y))

# 7. 矩阵-向量相乘 (矩阵乘以一个向量)
print('7.矩阵-向量相乘(向量积)')
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)  # 创建一个形状为 (5, 4) 的张量
x = torch.arange(4, dtype=torch.float32).reshape(4, 1)   # 创建一个形状为 (4, 1) 的张量（向量）
print('A:', A)  # 输出矩阵 A (5*4)
print('x:', x)  # 输出向量 x (4*1)
print('torch.mv(A, x):', torch.mv(A, x))  # 将矩阵 A 与向量 x 相乘，结果是一个 (5, 1) 的张量
# 矩阵-向量相乘：torch.mv(A, x) 执行矩阵 A 和向量 x 的乘法运算，返回结果是一个向量。

# 8. 矩阵-矩阵相乘 (两个矩阵相乘)
print('8.矩阵-矩阵相乘(矩阵积)')
B = torch.ones(4, 3)  # 创建一个形状为 (4, 3) 的全 1 矩阵
print('A:', A)  # 输出矩阵 A (5*4)
print('B:', B)  # 输出矩阵 B (4*3)
print('torch.mm(A, B):', torch.mm(A, B))  # 将矩阵 A 与矩阵 B 相乘，结果是一个 (5, 3) 的张量
# 矩阵-矩阵相乘：torch.mm(A, B) 执行矩阵 A 和矩阵 B 的乘法运算，返回结果是一个矩阵。

# 9. 范数 (向量和矩阵的范数)
print('9.范数')
u = torch.tensor([3.0, -4.0])  # 创建一个二维向量 u
print('向量的𝐿2范数:', torch.norm(u))  
# 计算向量 u 的 𝐿2 范数，即 Euclidean 范数（√(3^2 + (-4)^2) = 5）
print('向量的𝐿1范数:', torch.abs(u).sum())  
# 计算向量 u 的 𝐿1 范数，即所有元素绝对值的和 (3 + 4 = 7)

v = torch.ones((4, 9))  # 创建一个形状为 (4, 9) 的全 1 矩阵
print('v:', v)  # 输出矩阵 v
print('矩阵的𝐿2范数:', torch.norm(v))  
# 计算矩阵 v 的 #!(𝐿2 范数，即所有元素平方和的平方根)

# 10. 根据索引访问矩阵
print('10.根据索引访问矩阵')
y = torch.arange(10).reshape(5, 2)  # 创建一个形状为 (5, 2) 的张量
print('y:', y)  # 输出矩阵 y
index = torch.tensor([1, 4])  # 指定一个索引列表
print('y[index]:', y[index])  
# 根据索引访问矩阵 y 的行 (取出第 1 行和第 4 行)

# 11. 理解 PyTorch 中的 gather() 函数
print('11.理解pytorch中的gather()函数')
a = torch.arange(15).view(3, 5)  # 创建一个形状为 (3, 5) 的张量
print('11.1二维矩阵上gather()函数')
print('a:', a)  # 输出矩阵 a
b = torch.zeros_like(a)  # 创建一个与 a 形状相同的全零矩阵
b[1][2] = 1  # 给矩阵 b 的 (1, 2) 元素赋值为 1
b[0][0] = 1  # 给矩阵 b 的 (0, 0) 元素赋值为 1
print('b:', b)  # 输出矩阵 b

# 使用 gather() 函数在第 0 维和第 1 维上收集元素
c = a.gather(0, b)  # 按照 b 的值，从 a 的第 0 维度收集元素
d = a.gather(1, b)  # 按照 b 的值，从 a 的第 1 维度收集元素
print('d:', d)  # 输出 d

print('11.2三维矩阵上gather()函数')
a = torch.randint(0, 30, (2, 3, 5))  # 创建一个形状为 (2, 3, 5) 的随机整数张量
print('a:', a)  # 输出三维张量 a

index = torch.LongTensor([[[0, 1, 2, 0, 2],
                           [0, 0, 0, 0, 0],
                           [1, 1, 1, 1, 1]],
                          [[1, 2, 2, 2, 2],
                           [0, 0, 0, 0, 0],
                           [2, 2, 2, 2, 2]]])  # 指定一个形状为 (2, 3, 5) 的索引张量
print(a.size() == index.size())  # 检查 a 的尺寸是否与 index 相同 (应为 True)

b = torch.gather(a, 1, index)  # 在第 1 维上按照 index 收集 a 的元素
print('b:', b)  # 输出 b
c = torch.gather(a, 2, index)  # 在第 2 维上按照 index 收集 a 的元素
print('c:', c)  # 输出 c

index2 = torch.LongTensor([[[0, 1, 1, 0, 1],
                            [0, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1]],
                           [[1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [1, 1, 0, 0, 0]]])  # 另一个索引张量

d = torch.gather(a, 0, index2)  # 在第 0 维上按照 index2 收集 a 的元素
print('d:', d)  # 输出 d

# 12. 理解 PyTorch 中的 max() 和 argmax() 函数
print('12.理解pytorch中的max()和argmax()函数')
a = torch.tensor([[1, 2, 3], [3, 2, 1]])  # 创建一个 2*3 的张量
b = a.argmax(1)  # 返回每一行中最大元素的索引 (axis=1, 行)
c = a.max(1)  # 返回每一行的最大元素及其索引
d = a.max(1)[1]  # 仅返回每一行的最大元素的索引

print('a:', a)  # 输出 a
print('a.argmax(1):', b)  # 输出 b: [2, 0]，表示每一行最大值的索引
print('a.max(1):', c)  # 输出 c: 包含最大值及其索引的元组
print('a.max(1)[1]:', d)  # 输出 d: 仅索引

# 13. item() 函数
print('13.item()函数')
a = torch.Tensor([1, 2, 3])  # 创建一个 1*3 的浮点数张量
print('a[0]:', a[0])  # 直接通过索引访问 a 的第一个元素，返回的是 tensor 对象
print('a[0].item():', a[0].item())  # 使用 item() 方法获取纯 Python 数值 (1.0)
