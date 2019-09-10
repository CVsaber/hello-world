# -*- coding: utf-8 -*-
# @Time    : 2019/9/2 0:03
# @Environment: Windows10
# @Author  : moli99
# @Email   : guiczhang@163.com
# @CSDN    : https://blog.csdn.net/u014421797
# @File    : day2.py
# @Software: PyCharm
"""运算练习"""
import torch

x = torch.ones(5, 3)
y = torch.rand(5, 3)
print(x+y)  # 加法1
print(torch.add(x, y))  # 加法2
result = torch.empty(5, 3) # 给定一个输出张量参数
torch.add(x, y, out=result)
print(result) # 加法3

# 加法：原地原位操作(in-place)
# 任何一个in-place改变张量的操作后面都固定一个_。
# 例如x.copy_(y)、x.t_()将更改x
y.add_(x)
print(y)

# 索引
print(y[:, 1])

# 改变形状torch.view
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)   # -1表示从其他维度推断应该是几
# 例如这里（-1,8)表示从8推断，将x遍历完，保证列为8，则推出-1应该为2
print(x.size(), y.size(), z.size())
print(x, y, z)

# 如果是仅包含一个元素的tensor，可以使用.item()来得到对应的数值
x = torch.randn(1)
print(x)
print(x.item())

# tensor和numpy之间的转化,并且他们共享内存，改变一个，另一个也改变
# tensor转numpy
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
# 改变
a.add_(1)
print(a)
print(b)

# numpy转tensor
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a,'\n',b)

# cuda的使用.to
if torch.cuda.is_available():
    device = torch.device("cuda")
    y = torch.one_like(x, device = device)
    x = x.to(device)
    z =x + y
    print(z)
    print(z.to("cpu", torch.double))
