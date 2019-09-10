# -*- coding: utf-8 -*-
# @Time    : 2019/9/1 0:30
# @Environment: Windows10
# @Author  : moli99
# @Email   : guiczhang@163.com
# @CSDN    : https://blog.csdn.net/u014421797
# @File    : day1.py
# @Software: PyCharm

"""
pytorch学习
"""
import torch

x = torch.empty(5,3)    # 无初始化
print(x)

y = torch.rand(5,3)   # 随机初始化
print(y)

z = torch.zeros(5,3,dtype=torch.long) # long型
print(z)

w = torch.tensor([5.5, 3])  # 数据构造张量
print(w)

# 根据已有的tensor建立新的tensor。
# 除非用户提供新的值，否则这些方法将重用输入张量的属性，例如dtype
x = x.new_ones(5, 3, dtype=torch.double)
print(x)
x = torch.randn_like(x, dtype=torch.float) # 重载dtype
print(x)    # size是一样的

print(x.size()) # 张量的形状，本质是tuple