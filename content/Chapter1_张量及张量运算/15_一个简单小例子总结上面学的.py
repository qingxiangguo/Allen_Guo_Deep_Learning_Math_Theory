# _*_ coding=utf-8 _*_
import torch

tensor1 = torch.tensor([[5, 6]])  # 二维数组，shape为（1，2）
print(tensor1, tensor1.shape)
'''
tensor([[5, 6]]) torch.Size([1, 2])
'''
tensor2 = torch.tensor([[2], [9]]) # 二维数组，shape为（2，1）
print(tensor2, tensor2.shape)
'''
tensor([[2],
        [9]]) torch.Size([2, 1])
'''
# 由于tensor1的列，等于tensor2的行数，所以可以矩阵乘法，结果维度为，tensor1的行，等于tensor2的列，即(1,1)
print(torch.matmul(tensor1,tensor2))
'''
tensor([[64]])
'''

# 也可以用mm函数，结果一样
print(torch.mm(tensor1,tensor2))
'''
tensor([[64]])
'''

# 也可以使用对位相乘，torch.mul函数，两者形状不一样，但是(1,2)和(2,1)可以广播为(2,2)
print(torch.mul(tensor1,tensor2))
'''
tensor([[10, 12],
        [45, 54]])
'''