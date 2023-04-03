# _*_ coding=utf-8 _*_

"""
对于不互斥的多标签问题（即每个样本可以属于多个类别），我们通常在最后一层使用 sigmoid 激活函数。因为 sigmoid 函数可以将每个类别的输出限制在
0 到 1 之间，从而可以解释为每个类别的概率。同时，由于每个类别的预测是独立的，sigmoid 函数可以让每个类别之间的预测不受其他类别的影响。

在这种情况下，损失函数通常使用二元交叉熵损失（Binary Cross-Entropy Loss，简称 BCE）。与二分类问题相似，我们将逐元素地计算每个类别的
二元交叉熵损失，然后对所有类别求平均以得到最终的损失值。这样的损失函数鼓励模型在每个类别上都产生正确的概率预测，同时考虑到不同类别之间的独立性。

假设我们有一个简单的多标签分类问题，目标是根据电影的简介文本为电影分配多个类型标签。类型标签包括 "动作"、"喜剧" 和 "浪漫"，
一个电影可能同时属于这三个类型。

假设输入特征向量的长度为 50，即 input_dim = 50。输出有 3 个标签，即 output_dim = 3。

特征向量的长度为 50 意味着输入数据由 50 个数值组成，这 50 个数值是从原始输入数据（在本例中为电影简介文本）经过预处理得到的。
这些数值可以看作是原始数据的一种数值表示，有助于神经网络学习和理解输入数据的模式。

这 50 个数值（即特征）可能是各种各样的信息。例如，在处理文本时，我们可以使用词频、TF-IDF（词频-逆文档频率）
等方法将文本转换为固定长度的特征向量。
在这种情况下，这 50 个特征可能表示输入文本中最相关或最重要的 50 个单词的 TF-IDF 值。当然，实际应用中的特征向量长度可能远大于 50，
取决于问题的复杂性和数据的维度。

假设我们有以下电影简介文本：

"An action-packed thriller about a former intelligence agent who is drawn back into the world of espionage to stop a
dangerous terrorist plot."

为了将这段文本转换为一个特征向量，我们可以采用词频 (term frequency, TF) 或者词频-逆文档频率
(term frequency-inverse document frequency, TF-IDF) 等方法。这里我们以词频为例。

首先，我们需要构建一个词汇表，即根据我们的数据集中的所有电影简介提取出现频率较高的前 50 个单词。假设这些词汇为：

['action', 'adventure', 'agent', 'alien', 'back', 'battle', 'city', 'comedy', 'dangerous', 'detective',
'drama', 'earth', 'family', 'fight', 'friends', 'future', 'hero', 'intelligence', 'journey', 'life',
'love', 'mission', 'murder', 'mystery', 'new', 'old', 'past', 'police', 'power', 'quest', 'race',
'revenge', 'romance', 'save', 'sci-fi', 'secret', 'space', 'stop', 'story', 'struggle', 'survive',
'terrorist', 'thriller', 'time', 'war', 'world', 'young', 'zombie']

然后，我们计算这些词在电影简介中出现的次数，并将这些词频数值作为输入特征向量。例如，在这个简介中，"action" 出现了 1 次，
"agent" 出现了 1 次，"back" 出现了 1 次，"dangerous" 出现了 1 次，"intelligence" 出现了 1 次，"stop" 出现了 1 次，
"terrorist" 出现了 1 次，"thriller" 出现了 1 次，其他词汇均未出现。因此，我们得到如下特征向量：

[1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]

这个特征向量将被用作神经网络的输入。需要注意的是，这里我们采用了一个简化的例子，实际应用中的特征提取和表示方法可能会更加复杂和高维。

我们可以使用一个简单的全连接神经网络进行分类，网络结构如下：

输入层：50 个输入特征。
隐藏层：一个包含 100 个神经元的全连接层，使用 ReLU 激活函数。
输出层：一个包含 3 个神经元的全连接层，使用 sigmoid 激活函数。

现在，我们有一个输入样本 x，经过预处理得到一个长度为 50 的特征向量。我们将该向量输入神经网络，经过隐藏层的计算和激活，
然后通过输出层，使用 sigmoid 激活函数得到 3 个输出值。假设输出值为 [0.8, 0.1, 0.6]，表示该电影分别有 80% 的概率是动作片，
10% 的概率是喜剧片，以及 60% 的概率是浪漫片。

为了计算损失，我们需要真实的标签值。假设真实标签为 [1, 0, 1]，表示这部电影是动作片和浪漫片的组合。我们可以使用二元交叉熵
损失函数（BCE）计算损失，对每个类别逐元素地计算二元交叉熵损失，然后对所有类别求平均。在这个例子中，损失值可以通过如下公式计算：

BCE_loss = -(1 * log(0.8) + 0 * log(0.1) + 1 * log(0.6)) / 3

接下来，我们会使用优化器（如随机梯度下降）更新模型参数以减小损失。重复此过程多次，直到模型收敛。
"""