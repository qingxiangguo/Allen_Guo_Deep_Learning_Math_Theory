# _*_ coding=utf-8 _*_
# 张量属性包括，形状、数据类型以及存储设备位置（cpu还是gpu）
import torch

tensor = torch.rand(3, 4)  # 创建一个3行4列的二维张量

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

'''
tensor = torch.rand(3, 4)

Shape of tensor: torch.Size([3, 4])
Datatype of tensor: torch.float32
Device tensor is stored on: cpu
'''