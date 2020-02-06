# Writing Part

## Neural Machine Translation with RNN

### (g)

$$Softmax(x_i) = \frac{exp(x_i)}{\sum_jexp(x_j)}$$

如果 $x_j = -\infty$，那么 $exp(x_j) = 0$，所以 padding 的部分不会参与 Softmax 的计算（其 Softmax 也为 0），保证 attention 的 Softmax 得到正确的结果，不会受到 padding 的影响。

### (j)

点乘：
- 优点：速度快、比较简单；
- 缺点：得到的 attention score 可能不太准确。

additive attention 相比 multiplicative attention：
- 优点：信息更准确；
- 缺点：有更多超参数需要调整，参数需要更多存储空间，计算速度慢。