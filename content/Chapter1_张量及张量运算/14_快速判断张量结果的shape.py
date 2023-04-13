# _*_ coding=utf-8 _*_
'''
实际过程中很少需要你手动计算矩阵，特别是多维矩阵
在pytorch中，matrix dimensions指的是后两个维度
前面的，第三个维度，以及三维以上的维度，称为batch dimensions
在matmul计算有三维以上数组时，结果的维度，batch维度是由输入的batch维度广播，或者保留生成的，最后末尾两位是矩阵乘积的（行、列规则）决定的，或者是矩阵向量乘积决定的
也就是只有末尾两个位置，是真正参与计算得出的，前面位置靠扩展
比如两个维度分别是(5, 2, 3, 1)和(2,1,3)，先把第一个5拿出来，第二个2保留（因为两个一样），然后（3，1）和(1,3)进行矩阵乘法，得到（3，3）
所以最后结果shape为（5, 2, 3, 3）
'''
import torch
# 例子一，涉及到三维以上
tensor1 = torch.randn(5, 2, 3, 1) # 注意，这里batch dimensions有5和2，如果2不是2，也不是1，而是别的数字，那么这两个矩阵就不能广播了
tensor2 = torch.randn(2,1,3)
print(torch.matmul(tensor1, tensor2).size()) # torch.Size([5, 2, 3, 3])

# 例子二，仅仅是低维
tensor1 = torch.randn(3, 4)
tensor2 = torch.randn(4)
print(torch.matmul(tensor1, tensor2).size())  # torch.Size([3]) 矩阵乘向量，结果就是降一维，这是矩阵乘向量本身决定的

# 例子三 # batched matrix （就是三维及三维以上的张量） x broadcasted vector
tensor1 = torch.randn(10, 3, 4)
tensor2 = torch.randn(4)
print(torch.matmul(tensor1, tensor2).size()) # torch.Size([10, 3])
'''
首先把batch位拿出来不考虑，10，然后(3,4)和(4)，矩阵乘向量，结果就是降一维，为3，所以最后shape为（10，3）
'''

# 例子四batched matrix x batched matrix
tensor1 = torch.randn(10, 3, 4)
tensor2 = torch.randn(10, 4, 5)
print(torch.matmul(tensor1, tensor2).size())  # torch.Size([10, 3, 5])
# 三维数组，batch位dimension，也就是10必须一样，先拿出来，然后(3, 4) 再与(4,5)进行矩阵乘积得到（3, 5），所以最后维度为(10, 3, 5)

# 例子五，batched matrix x broadcasted matrix
tensor1 = torch.randn(10, 3, 4)
tensor2 = torch.randn(4, 5)
print(torch.matmul(tensor1, tensor2).size())  # torch.Size([10, 3, 5])
# 首先判断，形状不一样，那么需要广播，batch位dimension，上面数组有10，而下面没有，所以是可以广播的，广播时，10先拿出来，然后(3, 4) 再与(4,5)进行矩阵乘积得到（3, 5），所以最后维度为(10, 3, 5)

# 例子六，两个都是三维
tensor1 = torch.randn(2, 5, 3)
tensor2 = torch.randn(1, 3, 4)
print(torch.matmul(tensor1, tensor2).size()) # torch.Size([2, 5, 4])
# 首先判断，形状不一模一样，需要广播，那么能不能广播呢？batch位，一个是2，一个是1（是2也可以，但别的数字不行），满足广播条件
# 所以广播时，把2提取出来，然后(5, 3) 再与(3,4)进行矩阵乘积得到（5, 4）,所以最后的维度为（2，5，4）

# 例子七，涉及到四维
tensor1 = torch.randn(2, 1, 3, 4)
tensor2 = torch.randn(   5, 4, 2)
print(torch.matmul(tensor1, tensor2).size()) # torch.Size([2, 5, 3, 2])
# 首先判断，形状不一模一样，需要广播，那么能不能广播呢？第一个batch位，一个是2，一个是没有，第2个batch位，一个是5，一个是1，满足广播条件
# 所以广播时，把batch位置提取出来（下面那个填充为2），上面的1扩展为5，获得了（2，5）
# 然后(3, 4) 再与(4,2)进行矩阵乘积得到（3, 2）,所以最后的维度为（2，5，3，2）



