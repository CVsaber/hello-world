# -*- coding: utf-8 -*-
# @Time    : 2019/9/7 22:54
# @Environment: Windows10
# @Author  : moli99
# @Email   : guiczhang@163.com
# @CSDN    : https://blog.csdn.net/u014421797
# @File    : day5.py
# @Software: PyCharm
"""分类器"""
# 数据：numpy转tensor
# 特别的，对于图像任务，我们创建了一个包 torchvision，
# 它包含了处理一些基本图像数据集的方法。
# 这些数据集包括 Imagenet, CIFAR10, MNIST 等。
# 除了数据加载以外，torchvision 还包含了图像转换器，
# torchvision.datasets 和 torch.utils.data.DataLoader
import torch
import torchvision
