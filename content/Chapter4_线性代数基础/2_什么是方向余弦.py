# _*_ coding=utf-8 _*_
"""
在三角函数中，余弦（cos）等于邻边比斜边
如果你画出三维的坐标系中，向量(x, y, z)，以及他们的模L = √(x²+y²+z²)
可以发现，cos(α) = x / L
cos(β) = y / L
cos(γ) = z / L
其中，α，β，γ分别是这向量与三个坐标轴之间的角度

方向余弦，是一个向量的三个方向余弦分别是这向量与三个坐标轴之间的角度的余弦

题目中经常可以根据这个来求角度

例题：已知两点，M1 = (5, √2, 2), M2 = (4, 0, 3)，求：向量M1M2的模，方向余弦
方向角，并求与M1M2方向一致的单位向量

可以知道，向量M1M2 = M2 - 1 = (-1, -√2, 1)， 模等于，√(1+2+1) = 2
cos(α) = -1/2, cos(β) = -√2/2, cos(γ) = 1/2，所以可以求出角度

单位化，就是每个坐标除以模即可， 这样可以得到这个方向上的单位向量，长度为1
"""