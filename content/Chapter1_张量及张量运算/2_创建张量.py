# _*_ coding=utf-8 _*_
# 张量可以直接从数据中创建。数据类型是自动推断出来的
import torch
import numpy as np

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)  # 直接根据数据创建

# 也可以从NumPy数组中创建张量
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# 除非明确重写，否则新的张量保留了原张量的属性（形状、数据类型）
x_ones = torch.ones_like(x_data) # retains the properties of x_data
# one_like()方法是指，生成与input形状相同，元素全为1的张量
print(f"Ones Tensor: \n {x_ones} \n")

'''
Ones Tensor:   #是个二维张量，矩阵
 tensor([[1, 1],
        [1, 1]]) 

'''

x_rand = torch.rand_like(x_data, dtype=torch.float) # 重写x_data的数据类型为浮点型
# 返回输入形状相同大小的张量，该张量由区间[0,1)上均匀分布的随机数填充
print(f"Random Tensor: \n {x_rand} \n")

'''
Random Tensor: 
 tensor([[0.4815, 0.7297],
        [0.8659, 0.5418]]) 
'''

# 也可以根据shape参数来生成指定形状的数组
shape = (2, 5)
rand_tensor = torch.rand(shape)  # 生成2行5列的二维张量
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

'''
Random Tensor: 
 tensor([[0.1508, 0.6889, 0.6048, 0.5894, 0.7655],
        [0.2653, 0.7928, 0.1190, 0.3876, 0.1684]]) 

Ones Tensor: 
 tensor([[1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1.]]) 

Zeros Tensor: 
 tensor([[0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.]])
'''

