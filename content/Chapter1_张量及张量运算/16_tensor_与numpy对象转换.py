# _*_ coding=utf-8 _*_
'''
它将一个张量对象转换为numpy.ndarray对象
转换后的tensor与numpy指向同一地址，所以，对一方的值改变另一方也随之改变
'''

import torch
import numpy as np

t = torch.ones(5)
print(f"t: {t}")
n = t.numpy() # 转换为一个numpy对象
print(f"n: {n}")

''''
t: tensor([1., 1., 1., 1., 1.])
n: [1. 1. 1. 1. 1.]
'''

t.add_(1) # 改变原来的张量，也改变numpy
print(f"t: {t}")
print(f"n: {n}")

"""
t: tensor([2., 2., 2., 2., 2.])
n: [2. 2. 2. 2. 2.]
"""

# 同理numpy.ndarray对象也可以转换为张量对象
n = np.ones(7)
t = torch.from_numpy(n)

# NumPy数组中的变化反映在张量中
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")

"""
t: tensor([2., 2., 2., 2., 2., 2., 2.], dtype=torch.float64)
n: [2. 2. 2. 2. 2. 2. 2.]
"""