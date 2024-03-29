# _*_ coding=utf-8 _*_

"""
信息量的公式I(x)=-log2(p(x))是怎么推导出来的呢？为什么是Log?为什么是负数呢？ 首先假设我们不知道信息量定量的表达式，
以一个足球的例子来推导。f(x) = 信息量，f(阿根廷夺冠) = f(阿根廷进决赛) + f(阿根廷赢了决赛)。为什么这里是相加而不是相乘呢，
因为信息量是个量纲，是有单位的，不能相乘。假设阿根廷夺冠，阿根廷进决赛，阿根廷赢了决赛的概率分别是, 1/8, 1/2, 1/4

所以f(1/8) = f(1/4) + f(1/2)。为了保持自洽，信息量公式肯定要有log函数，这样log(a*b) = log(a) + log(b)。
此外，概率越小，信息量越多，又因为log是一个单调递增的函数，所以前面要加负号。至于底数是多少无所谓，一般取2，也就是一比特。

简单来说，信息量是衡量某一事件发生所带来的“惊讶程度”的量度，当概率较小的事件发生时，我们的惊讶程度较高，反之亦然。
信息量的公式I(x)=-log2(p(x))由以下几点考虑：

信息量与概率的关系：事件发生的概率越小，其信息量越大，因此信息量与概率应该是负相关的。

信息量的可加性：独立事件的信息量之和等于这些事件同时发生的信息量。在这里，信息量作为一个量纲，它是可加的。

对数函数的性质：对数函数具有log(a*b) = log(a) + log(b)的性质，这使得我们可以利用对数函数来满足信息量的可加性。

基于以上考虑，我们可以得出信息量公式I(x)=-log2(p(x))。负号的作用是确保信息量始终为正值，因为概率p(x)的取值范围在0到1之间。
使用对数函数是因为它满足可加性，并且它是单调递增的，能够保持概率与信息量之间的负相关关系。
至于底数，通常选择2，这样得到的信息量单位是比特（bit），当然，底数可以选择其他值，例如自然对数的底数e，此时信息量的单位是纳特（nat）。

"""