# _*_ coding=utf-8 _*_
'''
torch.bmm(input, mat2, *, deterministic=False, out=None) → Tensor
对 input 和 mat2 矩阵执行批处理矩阵积。
input 和 mat2 必须是三维张量，每个张量包含相同数量的矩阵。
输入：
input tensor 维度：(b×n×m) ;  input的列数m等于mat2的行数m
mat2 tensor 维度： (b×m×p) ;
输出：
out tensor 维度： (b×n×p) .
bmm是不支持广播的，批量矩阵(batch matrix)实际上指的就是一个三维张量，里面包括了多个二维矩阵
看函数名就知道，在torch.mm的基础上加了个batch计算，不能广播
'''
import torch

# torch.bmm例子一

matrix1 = torch.rand(2, 3, 3)
matrix2 = torch.rand(2, 3, 4)

print("First input matrix - \n",matrix1)
print("\nSecond input matrix - \n",matrix2)
print("\nResultant output matrix - \n",torch.bmm(matrix1, matrix2))

'''
First input matrix - 
 tensor([[[0.3260, 0.4180, 0.6664],
         [0.0429, 0.5822, 0.9498],
         [0.7134, 0.3711, 0.2579]],

        [[0.4906, 0.2309, 0.7525],
         [0.2104, 0.9539, 0.0038],
         [0.9404, 0.5632, 0.1236]]])

Second input matrix - 
 tensor([[[0.7997, 0.5746, 0.7233, 0.5779],
         [0.4383, 0.8301, 0.9459, 0.4800],
         [0.0629, 0.7329, 0.8953, 0.5679]],

        [[0.3831, 0.5112, 0.3799, 0.6605],
         [0.2302, 0.5020, 0.1581, 0.2143],
         [0.8493, 0.8762, 0.6936, 0.3452]]])

Resultant output matrix - 
 tensor([[[0.4858, 1.0227, 1.2278, 0.7675],
         [0.3492, 1.2041, 1.4321, 0.8437],
         [0.7495, 0.9071, 1.0980, 0.7369]],

        [[0.8802, 1.0260, 0.7448, 0.6333],
         [0.3034, 0.5897, 0.2333, 0.3446],
         [0.5950, 0.8718, 0.5320, 0.7845]]])
'''

# torch.bmm例子二
# 我们现在只关心，在计算过程中size的变化
BatchTensor1 = torch.randn(10, 3, 4)
BatchTensor2 = torch.randn(10, 4, 5)
resultingTensor = torch.bmm(BatchTensor1, BatchTensor2)
print(resultingTensor.size()) # torch.Size([10, 3, 5])

'''
当输入的第一个张量是(b * n * m)，第二个张量是(b * m * p)，
那么两个输入张量的矩阵乘法得到的结果矩阵将是(b * n * p)张量。
'''

# torch.mm和torch.bmm有什么区别呢？
# mm只用于两个矩阵都是2维的矩阵乘法，bmm只用于两个矩阵都是3维的矩阵乘法。
# mm两个矩阵的第一维不一定是相同的值，但是bmm两个输入矩阵的第一维必须是相同的值，否则无法完成计算
# 两者都不支持广播，不支持允许不同形状的张量的广播，较小的张量被广播给较大的张量以适合形状


