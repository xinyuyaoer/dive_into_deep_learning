import os

import numpy as np
import pandas as pd
import torch
from numpy import nan as NaN  # 将 NaN 引入命名空间，方便使用

# 创建上级目录中的 data 文件夹，如果已经存在则不会报错
os.makedirs(os.path.join('..', 'data'), exist_ok=True)  
datafile = os.path.join('..', 'data', 'house_tiny.csv')  # 定义 CSV 文件的路径
with open(datafile, 'w') as f:  # 打开文件准备写入数据，如果文件不存在则创建
    f.write('NumRooms,Alley,Price\n')  # 写入列名
    f.write('NA,Pave,127500\n')  # 写入第一行数据，'NA' 表示缺失值
    f.write('2,NA,106000\n')  # 写入第二行数据，'NA' 表示缺失值
    f.write('4,NA,178100\n')  # 写入第三行数据，'NA' 表示缺失值
    f.write('NA,NA,140000\n')  # 写入第四行数据，'NA' 表示缺失值

# 读取 CSV 文件为 Pandas DataFrame，可以看到 'NA' 被自动识别为 NaN
data = pd.read_csv(datafile)
print('1.原始数据:\n', data)

# 将数据分为输入部分 (前两列) 和输出部分 (最后一列)
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
# 用每一列的均值填充 NaN，注意这里只对数值列起作用
inputs = inputs.fillna(inputs.mean())
print(inputs)  # 打印填充后的输入数据
print(outputs)  # 打印输出数据

# 使用 Pandas 的 get_dummies 函数对离散值或类别值进行独热编码
# get_dummies 会将类别值转换为二进制特征，同时将 NaN 视作一种类别
inputs = pd.get_dummies(inputs, dummy_na=True)
print('2.利用 pandas 中的 get_dummies 函数处理:\n', inputs)

# 将 Pandas DataFrame 转换为 PyTorch 张量，方便在深度学习中使用
x, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
print('3.转换为张量：')
print(x)  # 打印输入张量
print(y)  # 打印输出张量

# 扩展讲解 fillna 的多种用法
df1 = pd.DataFrame([[1, 2, 3], [NaN, NaN, 2], [NaN, NaN, NaN], [8, 8, NaN]])  # 创建一个示例 DataFrame
print('4.函数 fillna 的用法：')
print(df1)  # 打印原始 DataFrame

# 用常数填充 NaN，默认不会修改原 DataFrame
print(df1.fillna(100))  

# 通过字典为不同列填充不同的值，{列索引: 填充值}
print(df1.fillna({0: 10, 1: 20, 2: 30}))  

# 使用前一个有效值来填充 NaN，若前面没有值则不填充
print(df1.fillna(method='ffill'))  

# 使用 inplace=True 直接修改原 DataFrame，而不是返回新对象
# print(df1.fillna(0, inplace=True))  # 被注释掉的代码，取消注释后可以直接修改 df1

# 创建一个随机 5x5 的 DataFrame
df2 = pd.DataFrame(np.random.randint(0, 10, (5, 5)))  
df2.iloc[1:4, 3] = NaN  # 将指定的索引处的值设置为 NaN
df2.iloc[2:4, 4] = NaN  # 在另一列中设置 NaN
print(df2)  # 打印带有 NaN 的 DataFrame

# 使用 'bfill' 方法（从后向前填充），限制最多填充 2 个 NaN
print(df2.fillna(method='bfill', limit=2))  

# 使用 'ffill' 方法（从前向后填充），沿列方向填充且限制为 1 个 NaN
print(df2.fillna(method="ffill", limit=1, axis=1))  


### 代码讲解：
""" 1. **文件操作**：
   - 代码首先创建了一个目录，然后生成了一个 CSV 文件，并向文件中写入了数据。文件包含三个字段：`NumRooms`（房间数量）、`Alley`（小巷类型）、`Price`（房价）。部分数据被写为 `'NA'` 表示缺失值。

2. **读取数据**：
   - 使用 `pd.read_csv` 读取 CSV 文件，Pandas 自动将 `'NA'` 识别为 `NaN`，表示数据缺失。这一操作生成了一个 DataFrame，用于后续的数据处理。

3. **填充缺失值**：
   - 对于数值列（例如 `NumRooms`），代码使用列的均值填充缺失值。这是处理缺失数据的常见方法之一，有助于防止数据的稀疏性影响后续分析。

4. **处理类别变量**：
   - 对于类别列（例如 `Alley`），使用 `get_dummies` 进行独热编码，将类别值转换为数值形式。这样可以将分类数据转换为可以输入到机器学习模型的数值格式。

5. **数据转换为张量**：
   - 最后，使用 `torch.tensor` 将 DataFrame 转换为 PyTorch 张量，这是为了将数据准备好供深度学习模型使用。

6. **`fillna` 高级用法**：
   - 展示了 `fillna` 函数的多种高级用法，例如用常数填充、用字典为不同列填充不同的值、使用前一个或后一个有效值填充，以及限制填充的数量。这些技巧在处理实际数据集时非常有用，特别是当数据具有复杂的缺失模式时。

这些操作是数据预处理的核心内容，尤其是在准备数据供机器学习或深度学习模型使用时。了解这些技术将帮助你更好地处理和分析数据。
 """


#! 补充：iloc 是 Pandas 库中的一种用于数据选择的函数，它的名字来源于 "integer location" 的缩写。iloc 允许你通过行号和列号来选择数据，基于数据的整数索引进行访问，而不是基于标签（如列名或索引名）。
"""
### `iloc` 的基本用法

1. **选择行**：
   - `df.iloc[i]`：选择第 `i` 行的数据。`i` 是从 0 开始的整数索引。

2. **选择列**：
   - `df.iloc[:, j]`：选择第 `j` 列的数据。`j` 是从 0 开始的整数索引。`:` 表示选择所有行。

3. **选择特定的行和列**：
   - `df.iloc[i, j]`：选择第 `i` 行，第 `j` 列的数据。

4. **选择多个行和列**：
   - `df.iloc[i1:i2, j1:j2]`：选择从第 `i1` 行到第 `i2-1` 行，从第 `j1` 列到第 `j2-1` 列的子集。

### 示例

假设有一个 DataFrame：

```python
import pandas as pd

data = {
    'A': [1, 2, 3, 4],
    'B': [5, 6, 7, 8],
    'C': [9, 10, 11, 12],
    'D': [13, 14, 15, 16]
}
df = pd.DataFrame(data)
```

这个 DataFrame `df` 看起来是这样的：

|   | A | B | C | D |
|---|---|---|---|---|
| 0 | 1 | 5 | 9 | 13 |
| 1 | 2 | 6 | 10 | 14 |
| 2 | 3 | 7 | 11 | 15 |
| 3 | 4 | 8 | 12 | 16 |

- **选择第 2 行的所有数据**：`df.iloc[2]` 返回 `[3, 7, 11, 15]`。
- **选择第 2 列的所有数据**：`df.iloc[:, 2]` 返回 `[9, 10, 11, 12]`。
- **选择第 1 行，第 3 列的数据**：`df.iloc[1, 2]` 返回 `10`。
- **选择第 1 到第 3 行，第 2 到第 4 列的子集**：`df.iloc[1:3, 2:4]` 返回一个子 DataFrame：

  |   | C  | D  |
  |---|----|----|
  | 1 | 10 | 14 |
  | 2 | 11 | 15 |

### 总结
`iloc` 是 Pandas 中通过位置（整数索引）来定位和选择数据的工具。它非常适合在不使用标签时通过行列的具体位置提取数据。
"""