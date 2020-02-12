# Assignment 5

## Character-based convolutional encoder for NMT

### (a)

字母包含的信息量少于整个单词包含的信息量。

### (b)

对于 character-based embedding:
$$
E_{char} = e_{char} \times V_{char} \\
W_{conv} = e_{char} \times \text{kernel\_size( = k)} \times \text{channels( = f)}\\
b_{conv} = \text{channels( = f)} \\
\text{Notice : f is set to } e_{word} \text{ in this application} \\
W_{proj} = e_{word} \times e_{word} \\
W_{gate} = e_{word} \times e_{word} \\
b_{proj} = e_{word} \\
b_{gate} = e_{word}
$$

对于 word-based embedding:
$$
E_{word} = e_{word} \times V_{word} \\
$$

计算后得到 character-based 模型的参数量小于 word-based 模型。

### (c)

RNN 需要遍历两遍输入字母序列，而 CNN 只需要遍历一遍输入序列，并且卷积的计算有高效实现算法，运算只有乘加，比较简单，而 RNN 的计算相对比较复杂。

### (d)

max-pooling 的优点：运算较快，提取最重要的元素和信息，不容易受到噪声干扰；反向传播时梯度只回传给最重要的元素。

average-pooling 的优点：能够综合利用周围元素的信息，梯度回传时会传给所有的元素。

## Analyzing NMT Systems

### (a)

traducir, traduce 出现在词典中。

traduzco, traduces, traduzca, traduzcas 没有出现在词典中。

使用 word-based NMT 没有出现在词典中的词在词嵌入阶段会被嵌入为 \<UNK\>，character-based NMT 对没出现过的单词也可以利用构成字母相似的单词可能具有相近的意思对其进行词嵌入。

### (b)

| word        | nearest neighbor Word2Vec | nearest neighbor CharCNN |
| ----------- | ------------------------- | ------------------------ |
| financial   | economic                  | vertical                 |
| neuron      | nerve                     | Newton                   |
| Francisco   | san                       | France                   |
| naturally   | occurring                 | practically              |
| expectation | norms                     | expection                |

Word2Vec 主要是从词语的含义方面建模相似性，而 CharCNN 主要是根据构成单词的字母序列的相似程度建模相似性。

Word2Vec 主要用在语言模型等需要词语含义的情景下，例如 Skip-Gram 等预测可能出现的单词，而 CharCNN 是要在不知道任何上下文信息的情况下进行词嵌入，它不需要知道单词的含义，只捕捉字母层面的信息。 

## Note

CrossEntropyLoss 的使用，input = size(N, C), 最后一维是类别数，之后会计算出属于各个类别的概率，target = size(N)，只需要给出每个样本对应的正确的类别。 