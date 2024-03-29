# _*_ coding=utf-8 _*_

"""
层数越深，理论上拟合函数的能力增强，效果按理说会更好，但是实际上更深的层数可能会带来过拟合的问题，同时也会增加训练难度，使模型难以收敛。
因此我的经验是，在使用BP神经网络时，最好可以参照已有的表现优异的模型，
如果实在没有，则根据上面的表格，从一两层开始尝试，尽量不要使用太多的层数。

在CV、NLP等特殊领域，可以使用CNN、RNN、attention等特殊模型，不能不考虑实际而直接无脑堆砌多层神经网络。
尝试迁移和微调已有的预训练模型，能取得事半功倍的效果。

当你建立一个神经网络的模型时，隐藏层数的选择是很重要的。如果隐藏层数太少，模型的表现可能很差，因为模型不够复杂，
无法捕捉数据中的复杂关系。反之，如果隐藏层数太多，模型容易过拟合，导致在新数据上的表现很差。

举个例子，假如你要建立一个识别手写数字的模型，隐藏层数可能是1-2层。你可以试着训练一个1层的模型，
如果表现不好，再加一层。如果加了一层后，准确率得到了提高，但是随着训练数据量的增加，模型容易过拟合，这时候，可以使用【正则化】
技巧来防止过拟合。

同样的，隐藏层中的神经元数量也很重要。如果神经元数量太少，模型可能不够复杂，如果神经元数量太多，
模型可能容易过拟合。通常，我们会使用网格搜索或交叉验证的方法，来选择合适的神经元数量。

我们可以把深度学习比喻为一个构造模型的过程。你要建一个房子，那么你可以想象：如果没有隐藏层，那么房子只有一层
，它只能容纳一些最基本的东西，不能容纳更复杂的功能，就好像一个线性的函数。

如果房子有一层隐藏层，那么它就能容纳更多的功能了，就像从一个有限空间到另一个有限空间的映射。

如果房子有两层隐藏层，那么它就能通过配合适当的激活出输出，更好的表示决策边界，更好的拟合平滑映射。

如果房子有更多的隐藏层，那么它就能存储更多的复杂的描述，它们可以像一个自动特征工程，自动的学习一些特征，
但这也会带来问题，因为它有可能出现过拟合。

简而言之，在选择模型的时候，我们不能盲目的选择多层隐藏层，我们可以参考一些成熟的模型，或者是从一层或两层开始，
然后逐渐增加隐藏层，直到达到理想的效果

【过拟合】

过拟合是指模型在训练数据上表现得非常出色，但在预测新的未知数据时表现得不如预期。
这是因为隐藏层过多使得模型学习了训练数据的细节，
从而造成了高度的专业化，这些细节在新的未知数据中并不适用。因此，模型的泛化能力变差，导致了预测结果的误差增大。

深度学习隐藏层过多造成过拟合，这个概念可以用一个简单的比喻来解释。假设你是一个学生，你在做一份数学作业。
这份作业只需要算几道简单的数学题，但你却做了很多不必要的练习题。这些练习题是在让你掌握数学技能，但这些练习题不是数学作业题目的重点。

在机器学习的领域，隐藏层的数量也是一样的。如果模型的隐藏层数量太多，模型就像做了多余的练习题
，开始过度学习训练数据。结果就是，模型会忽略真实的数据特征，不能正确的预测新的数据。这种情况就是过拟合，就像是学生做了多余的练习题，而没有重点去学习数学知识。

解决过拟合的方法，有很多。一种方法是减少隐藏层的数量，让模型学习必要的特征。还有一种方法是使用正则化，
强制模型只学习必要的特征。另一种方法是使用更多的数据来训练模型，这样模型就不会过度学习训练数据。

【欠拟合】

深度学习中的欠拟合，可以用一个图画来理解。如果把学习过程比作在绘画一个函数图像，将数据分为两类：训练数据和测试数据。
我们希望通过绘制一条曲线，能够很好地表示训练数据，并且能够很好地适用于测试数据。

如果我们画的曲线很简单，不能很好地拟合数据，那么就是欠拟合的情况。这种情况下，训练误差可能很大，
而测试误差也可能很大。换句话说，我们的模型没有学会数据中的规律，对新数据的预测不够准确。

解决欠拟合的方法有两个：

增加模型的复杂度，例如增加更多的隐藏层，增加更多的神经元。
改变模型的结构，例如使用不同的激活函数，使用更复杂的网络结构。
但是，如果模型太复杂，也可能出现过拟合的情况，需要适当调整模型的复杂度。

模型的复杂度越高，它可以存储的信息和解释数据的能力就越强。当模型过于简单时，
它可能不能完整地存储和解释数据的所有关系和规律，从而造成欠拟合的情况。因此，增加模型的复杂度，
可以让模型更加灵活，能够更好地捕捉数据的关系，从而解决欠拟合的问题。
当然，如果增加的复杂度过大，可能会造成过拟合，所以需要合理选择模型的复杂度。

【总结】
机器学习中一个重要的话题便是模型的泛化能力,泛化能力强的模型才是好模型，
对于训练好的模型，若在训练集表现差，不必说在测试集表现同样会很差， 这可能是欠拟合导致；若模型在训练集表现非常好,却在测试集上差强人意，
则这便是过拟合导致的，过拟合与欠拟合也可以用Bias与 Variance的角度来解释，欠拟合会导致高Bias ,
过拟合会导致高Variance ,所以模型需要在Bias 与Variance之间做出一个权衡。

使用简单的模型去拟合复杂数据时，会导致模型很难拟合数据的真实分布，这时模型便欠拟合了, 或者说有很大的Bias,
Bias即为模型的期望输出与其真实输出之间的差异；有时为了得到比较精确的模型而过度拟合训练数据，或者模型复杂度过高时，
可能连训练数据的噪音也拟合了，导致模型在训练集上效果非常好，但泛化性能Q却很差，这时模型便过拟合了,
或者说有很大的Variance, 这时模型在不同训练集上得到的模型波动比较大，
Variance刻画了不同训练集得到的模型的输出与这些模型期望输出的差异。
"""