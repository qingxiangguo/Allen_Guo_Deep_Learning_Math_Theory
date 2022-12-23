# _*_ coding=utf-8 _*_
'''
复合函数求导，为从外向内，一层一层的求导，最外面的，乘以里面的，里面的再乘以最里面的
遇到x^(lnx)这种高阶函数，有两个函数乘积的，怎么处理呢？使用莱布尼茨公式
莱布尼兹公式，也称为乘积法则，是数学中关于两个函数的积的导数的一个计算法则
即(x.lnx)' = x'*lnx+lnx'*x，即交叉导数相加
此外还有商的求导公式：(v/u)'= (v'u-vu')/u^2
举一个例子，求y=2^(x/lnx)的导数，首先根据复合函数公式，看成y=2^x，因为a^x的导数为a^x*lna
y'= 2^(x/lnx)*ln2*(x/lnx)'
再根据商的求导公式，(x/lnx)'=(x'*lnx-(lnx)'*x)/(lnx)^2
上面的值等于：(lnx-1)/(lnx)^2

再来一个例子：y=(1+2x)^x次方如何求导
根据对数恒等式y=e^(ln((1+2x)^x))，利用自然对数变形，也叫洛必达法则
然后根据对数的性质，变形为y=e^(x*(ln(1+2x)))，就是简化了,可以用乘法莱布尼茨法则
y'=e^(x*(ln(1+2x)))*(x*(ln(1+2x)))'
y'=e^(x*(ln(1+2x)))*[1*ln(1+2x)+(ln(1+2x))'*x]
最后就变成求ln(1+2x)的导数了，(ln(1+2x))'=1/(1+2x)*2，反正就是一层层的来
'''