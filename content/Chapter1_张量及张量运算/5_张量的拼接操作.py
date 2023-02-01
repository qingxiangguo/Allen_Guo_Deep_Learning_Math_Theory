# _*_ coding=utf-8 _*_
import torch

tensor = torch.ones(4, 4) #创建一个4行4列的全1张量（二维）

tensor[:,1] = 0  # 所有行，第二列变为0， 逗号前是第一个维度指令，:是从头：到尾：步长为1的缩写；逗号后是第二个维度指令

# 在给定维度上，将几个不同张量拼接
t1 = torch.cat([tensor, tensor, tensor], dim=1) # 按第二个维度拼接，也就是按列，横着拼接
print(t1)

'''
tensor([[1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.]])
'''

# 也就是对于二维数组，有两个维度，0指第一个维度，也就是行，竖着拼接；1指第二个维度，也就是列，横着拼接
# 再来一组例子
A = torch.ones(2,3) # 2X3的二维数组
print('A = ', A)

B = 2*torch.ones(4,3)
print('B = ', B)

C = torch.cat([A, B], dim=0)  # 把A,B 按行拼接，也就是竖着
print('C = ', C)

D = 2*torch.ones(2,4)
print('D = ', D)

E = torch.cat([A, D], dim = 1)
print('E = ', E)

'''
A =  tensor([[1., 1., 1.],
        [1., 1., 1.]])
B =  tensor([[2., 2., 2.],
        [2., 2., 2.],
        [2., 2., 2.],
        [2., 2., 2.]])
C =  tensor([[1., 1., 1.],
        [1., 1., 1.],
        [2., 2., 2.],
        [2., 2., 2.],
        [2., 2., 2.],
        [2., 2., 2.]])
D =  tensor([[2., 2., 2., 2.],
        [2., 2., 2., 2.]])
E =  tensor([[1., 1., 1., 2., 2., 2., 2.],
        [1., 1., 1., 2., 2., 2., 2.]])
'''

print(D.shape) # torch.Size([2, 4])