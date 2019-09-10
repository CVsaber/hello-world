# -*- coding: utf-8 -*-
# @Time    : 2019/9/2 23:57
# @Environment: Windows10
# @Author  : moli99
# @Email   : guiczhang@163.com
# @CSDN    : https://blog.csdn.net/u014421797
# @File    : day3.py
# @Software: PyCharm
"""Autograd:自动求导"""

import torch
# 创建张量并追踪它的所有计算
x = torch.ones(2, 2, requires_grad=True)
print(x)
# 做一次计算
y = x + 2
print(y)
# 结果：tensor([[3., 3.],
#         [3., 3.]], grad_fn=<AddBackward0>)
# 发现有grad_fn属性，因为他是程序自己创建的tensor
print(y.grad_fn)

# 对y进行操作
z = y*y*3
out = z.mean()
print(z, '\n', out)
# tensor([[27., 27.],
#         [27., 27.]], grad_fn=<MulBackward0>)
#  tensor(27., grad_fn=<MeanBackward1>)
# .requires_grad_(...) 原地改变了现有张量的 requires_grad 标志。
# 如果没有指定的话，默认输入的这个标志是 False,可以看到上述结果没有grad属性
# 例子
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b)
# False
# True
# tensor(31.1965, grad_fn=<SumBackward0>)

# 反向传播
# out 是一个标量，因此 out.backward() 和 out.backward(torch.tensor(1.))
out.backward()
# 反向传播最后传到x
print(x.grad)


# autograd
x = torch.randn(3, requires_grad=True)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2
print(y)

gradients = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(gradients)
print(x.grad)

# 如果.requires_grad=True但是你又不希望进行autograd的计算，
# 那么可以将变量包裹在 with torch.no_grad()中:
print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)