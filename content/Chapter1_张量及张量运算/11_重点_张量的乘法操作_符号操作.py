# _*_ coding=utf-8 _*_
'''
当在张量运算的过程中，仅仅是使用*符号的话，含义是不同的，具体运算要取决于输入的张量，或者标量值
'''
import torch

# 第一种情况，二维矩阵和0维标量相乘
a = torch.tensor([[1, 3],  # a是3行，2列
                  [2, 3],
                  [4, 5]])

print(5*a)

'''
tensor([[ 5, 15],
        [10, 15],
        [20, 25]])
'''

# 第二种情况，一维向量（矢量）和一维向量（矢量）相乘
x = torch.tensor([1,2])  # 一维向量
y = torch.tensor([3,4])   # 一维向量
print(x*y)  # 等价于 print(torch.mul(x, y))

'''
tensor([3, 8])  # 输出一维张量间的对位相乘
'''

# 第三种情况，0维标量和一维向量（矢量）相乘
x = torch.tensor([1,2,3,4])
print(5*x)

'''
tensor([ 5, 10, 15, 20])  # 还是输出一维张量间
'''

# 第四种情况，二维数组和一维向量（矢量）相乘，本质还是对位相乘

a = torch.tensor([[1, 3],  # a是3行，2列
                  [4, 3],
                  [4, 5]])

x = torch.tensor([1,2])  # 一维向量，其个数必须与二维数组的列数相同，否则不行，会报错

print(a*x)

'''
tensor([[ 1,  6],
        [ 4,  6],
        [ 4, 10]])
'''

# 第五种情况，二维数组和二维数组相乘，但是两者形状一模一样，直接对位相乘，这也是最简单情况
a = torch.tensor([[1, 3],  # a是3行，2列
                  [4, 3],
                  [4, 5]])

b = torch.tensor([[2, 3],  # a是3行，2列
                  [2, 0],
                  [3, 5]])

print(a*b)

'''
tensor([[ 2,  9],  # 输出对位相乘
        [ 8,  0],
        [12, 25]])
'''

# 第六种情况，二维数组和二维数组相乘，两者不是一模一样，无法点乘，也无法进行维度的扩展
a = torch.tensor([[1, 3],  # a是4行，2列
                  [5, 3],
                  [4, 5],
                  [7, 8]])

b = torch.tensor([[1, 3, 1, 1],  # a是2行，4列
                  [4, 0, 5, 3]])

# print(a*b) a和b是无法进行点乘的

# 第七种情况，二维数组和二维数组相乘，但是两个数组可以扩展为相同维度，也就是可以利用广播机制变成(3, 4)

a = torch.tensor([[1., 1., 1., 1.],   # 3行4列  (3, 4)
                  [1., 2., 1., 1.],
                  [2., 1., 1., 1.]])

b = torch.tensor([[1],   # 3行1列  (3, 1)
                  [2],
                  [3],])

print(a*b)

'''
tensor([[1., 1., 1., 1.],
        [2., 4., 2., 2.],
        [6., 3., 3., 3.]])
'''

# 第八种情况，两个矩阵进行矩阵乘法
a = torch.tensor([[1, 3],  # a是4行，2列
                  [5, 3],
                  [4, 5],
                  [7, 8]])

b = torch.tensor([[1, 3, 1, 1],  # a是2行，4列
                  [4, 0, 5, 3]])

print(a@b)

'''
tensor([[13,  3, 16, 10],
        [17, 15, 20, 14],
        [24, 12, 29, 19],
        [39, 21, 47, 31]])
'''