# -*- coding: utf-8 -*-
# @Time    : 2019/9/5 23:38
# @Environment: Windows10
# @Author  : moli99
# @Email   : guiczhang@163.com
# @CSDN    : https://blog.csdn.net/u014421797
# @File    : day4.py
# @Software: PyCharm
"""神经网络"""
"""
典型训练过程：
1.定义一些包含可学习的参数（权重）
2.在数据集上迭代
3.神经网络处理输入
4.计算损失
5.梯度反向传播
6.更新参数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义网络LeNet
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1个image输入
        # 6个通道输出
        # 5*5卷积
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # y = Wx + b
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """前向传播"""
        # max pooling (2,2)
        # 卷积-激活-池化
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 池化窗为方的，可以指定一个参数
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # 扁平化处理
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        """二维特征扁平化"""
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)

# 在模型中必须要定义 forward 函数，
# 而backward 函数（用来计算梯度）会被autograd自动创建。
# 可以在 forward 函数中使用任何针对 Tensor 的操作。

# net.parameters()返回可被学习的参数（权重）列表和值
params = list(net.parameters())
print(len(params))
print(params[0].size()) # conv1的W

"""
测试网络代码随机输入32*32
"""
# torch.nn只支持小批量输入。
# 整个torch.nn包都只支持小批量样本，而不支持单个样本。
# 例如，nn.Conv2d接受一个4维的张量，
# 每一维分别是sSamples * nChannels * Height * Width（
# 样本数*通道数*高*宽）。
# 如果你有单个样本，只需使用input.unsqueeze(0)来添加其它的维数
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)
# 所有梯度缓存清零，随机梯度反向传播
net.zero_grad()
out.backward(torch.randn(1, 10))

# 均方误差损失 nn.MSELoss
output = net(input)
target = torch.randn(10) # 随机目标值
target = target.view(1, -1) # 使其与输出维度相同
criterion = nn.MSELoss()
loss = criterion(output, target)
print(loss)

# 后退几步
print(loss.grad_fn) # MESLoss
print(loss.grad_fn.next_functions[0][0]) #liner

# 反向传播
# 调用loss.backward()获得反向传播的误差。
# 但是在调用前需要清除已存在的梯度，否则梯度将被累加到已存在的梯度。
net.zero_grad()
print('卷积1在反向传播前的梯度')
print(net.conv1.bias.grad)
loss.backward() # 反向传播
print('卷积1在反向传播后的梯度')
print(net.conv1.bias.grad)

# 权重更新
import torch.optim as optim

# 优化
optimizer = optim.SGD(net.parameters(), lr=0.01)
# 训练
optimizer.zero_grad() # 梯度清零
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step() # 更新


