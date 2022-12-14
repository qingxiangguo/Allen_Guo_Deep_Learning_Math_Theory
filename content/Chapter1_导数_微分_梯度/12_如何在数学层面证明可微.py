# _*_ coding=utf-8 _*_
'''
例题一： 已知二元函数，f(x, y)， 当(x, y) = 0, 0 时，值 = 0； 当(x, y) ！= 0, 0时
函数值 = sin(xy)/√(x²+y²)， 那么在(0, 0)处，f(x, y): ( 选择题 )

A: 极限存在但不连续
B: 连续，但在(0, 0)处，函数关于x和y的偏导数均不存在（不可偏导）
C: 连续，在(0, 0)处，函数关于x和y的偏导数均存在， 但不可微
D:连续，在(0, 0)处，函数关于x和y的偏导数均存在， 且可微

解： 先看A,

首先求x ->0, y ->0处，函数的极限，limit x ->0, y ->0, sin(xy)/√(x²+y²)
由于第一重要极限，limit x ->0, y ->0, sin(xy) = limit x ->0, y ->0, xy
所以原式子等于，limit x ->0, y ->0, xy/√(x²+y²)

使用夹逼定理，xy/√(x²+y²)加绝对值， 又因为 x²+y² >= |2xy|

所以 0 <= limit x ->0, y ->0, |xy/√(x²+y²)| <= limit x ->0, y ->0, |xy|/√|2XY| = √(xy)/√2 = 0
左边右边都被0夹着，所以这个函数在0, 0处的极限存在，等于0，且等于函数值，0的，因此这个函数极限存在，且连续

再看偏导数是否存在，f(0, 0)处关于x的偏导数 = limit Δx -> 0, [f(Δx, 0) - f(0, 0)]/Δx
因为f(Δx, 0) = 0， f(0, 0)也等于0， 所以f(0, 0)处关于x的偏导数存在且等于0，由于f(x, y)中，x, y都是对称的
所以同理，所以f(0, 0)处关于y的偏导数存在且等于0

再看是否可微
根据可微的定义，Δz = AΔx + BΔy + o(√(Δx²+Δy²))， 移项，Δz - AΔx - BΔy = o(√(Δx²+Δy²))
由于A, B分别为偏导，所以原式等于
相当于要证明 Δz - (x偏导)Δx - (y偏导）Δy 的结果是 √(Δx²+Δy²)的高阶无穷小
根据高阶无穷小的定义，lim a/b = 0，那么a是比b高阶的无穷小，也就是要证明，limit Δx ->0, Δy ->0, (Δz - AΔx - BΔy)/√(Δx²+Δy²) = 0
由于已经证明偏导数，A， B都等于0，所以
要证明limit Δx ->0, Δy ->0, (Δz)/√(Δx²+Δy²) = 0
limit Δx ->0, Δy ->0, (f(Δx, Δy) - f(0, 0))/√(Δx²+Δy²) = 0
limit Δx ->0, Δy ->0, f(Δx, Δy)/√(Δx²+Δy²) = 0
将f(Δx, Δy)带入原式，变成limit Δx ->0, Δy ->0, sinx(Δx*Δy)/Δx²+Δy² = 0
令Δx = x, Δy = y, 变成求limit x ->0, y ->0， sin(xy)/(x²+y²)的极限
令y = kx方向逼近，原式变成，sin(kx²)/k²x²+x²，又由于第一重要极限
变成 1/1+k²，可就爱你这个极限不同逼近方式，结果不一样
所以limit Δx ->0, Δy ->0, (Δz)/√(Δx²+Δy²) 并不等于 0
所以不可微，因此答案选：C
'''