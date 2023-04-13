# _*_ coding=utf-8 _*_
"""
in-place operation在pytorch中是指改变一个tensor的值的时候，不经过复制操作，
而是直接在原来的内存上改变它的值。可以称之为“原地操作符”
可以节省内存
"""
import torch
a = torch.ones(3, 4)
print(a)

a.add_(5) # 原位加5

print(a)

"""
tensor([[1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.]])
tensor([[6., 6., 6., 6.],
        [6., 6., 6., 6.],
        [6., 6., 6., 6.]])
"""