import os

import numpy as np
import pandas as pd
import torch
from numpy import nan as NaN

os.makedirs(os.path.join('..', 'data'), exist_ok=True)  # 在上级目录创建data文件夹
datafile = os.path.join('..', 'data', 'house_tiny.csv')  # 创建文件
with open(datafile, 'w') as f:  # 往文件中写数据
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 第1行的值
    f.write('2,NA,106000\n')  # 第2行的值
    f.write('4,NA,178100\n')  # 第3行的值
    f.write('NA,NA,140000\n')  # 第4行的值

data = pd.read_csv(datafile)  # 可以看到原始表格中的空值NA被识别成了NaN
print('1.原始数据:\n', data)

inputs, outputs = data.iloc[:, 0: 2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())  # 用均值填充NaN
print(inputs)
print(outputs)
# 利用pandas中的get_dummies函数来处理离散值或者类别值。
# [对于 inputs 中的类别值或离散值，我们将 “NaN” 视为一个类别。] 由于 “Alley”列只接受两种类型的类别值 “Pave” 和 “NaN”
inputs = pd.get_dummies(inputs, dummy_na=True)
print('2.利用pandas中的get_dummies函数处理:\n', inputs)

x, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
print('3.转换为张量：')
print(x)
print(y)

# 扩展填充函数fillna的用法
df1 = pd.DataFrame([[1, 2, 3], [NaN, NaN, 2], [NaN, NaN, NaN], [8, 8, NaN]])  # 创建初始数据
print('4.函数fillna的用法：')
print(df1)
print(df1.fillna(100))  # 用常数填充 ，默认不会修改原对象
print(df1.fillna({0: 10, 1: 20, 2: 30}))  # 通过字典填充不同的常数，默认不会修改原对象
print(df1.fillna(method='ffill'))  # 用前面的值来填充
# print(df1.fillna(0, inplace=True))  # inplace= True直接修改原对象

df2 = pd.DataFrame(np.random.randint(0, 10, (5, 5)))  # 随机创建一个5*5
df2.iloc[1:4, 3] = NaN
df2.iloc[2:4, 4] = NaN  # 指定的索引处插入值
print(df2)
print(df2.fillna(method='bfill', limit=2))  # 限制填充个数
print(df2.fillna(method="ffill", limit=1, axis=1))  #

""" 这个代码主要演示了以下几个数据处理和转换的知识点，特别是在处理缺失数据和将数据转换为张量方面。具体来说，代码展示了以下内容：

1. **文件操作与数据读取**：
    - 使用 `os.makedirs` 创建目录，并用 `open` 函数写入 CSV 文件。这部分是基础的文件操作，目的是创建一个示例数据集以供后续操作。
    - 使用 `pd.read_csv` 读取 CSV 文件，并将其内容加载为 Pandas DataFrame。这是常见的数据读取方法，尤其是处理表格数据时。

2. **处理缺失数据**：
    - **`NaN` 填充**：
        - 使用 `fillna` 函数处理缺失数据（NaN）。代码通过两种方式填充 NaN：用列的均值填充（`inputs.fillna(inputs.mean())`）和用常数或前后值填充（如 `fillna(100)`、`fillna(method='ffill')`）。
    - 这部分知识点很重要，尤其是在实际数据分析中，缺失数据是很常见的问题。不同的填充方法适用于不同的情况。

3. **类别值处理**：
    - 使用 `pd.get_dummies` 函数将类别变量转换为独热编码（one-hot encoding）。这对机器学习中的数据预处理非常重要，因为模型往往不能直接处理字符串或类别型数据。
    - 代码中特别提到了如何将 `NaN` 作为一个独立的类别处理，这种处理方式适用于某些情况下 NaN 具有特定意义的场景。

4. **数据转换为张量**：
    - 最后，使用 `torch.tensor` 将 Pandas DataFrame 转换为 PyTorch 张量。这是为了将数据输入到深度学习模型中，因为 PyTorch 的模型接受张量作为输入。
    - 这个步骤展示了从数据预处理到深度学习模型准备的一个完整流程。

5. **`fillna` 函数的高级用法**：
    - 展示了 `fillna` 函数的不同用法，如用常数、字典、前后值填充以及限制填充次数或按列填充。这部分内容可以帮助你理解如何灵活地处理不同数据中的缺失值。

综上所述，代码讲解了数据处理中的多个重要知识点，包括缺失数据处理、类别值编码、数据格式转换，以及如何利用 PyTorch 进行深度学习数据的准备。这些都是数据科学与机器学习中非常实用的技能。
 """