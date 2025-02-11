# Writing Part

## Neural Machine Translation with RNN

### (g)

$$Softmax(x_i) = \frac{exp(x_i)}{\sum_jexp(x_j)}$$

如果 $x_j = -\infty$，那么 $exp(x_j) = 0$，所以 padding 的部分不会参与 Softmax 的计算（其 Softmax 也为 0），保证 attention 的 Softmax 得到正确的结果，不会受到 padding 的影响。

### (j)

点乘：
- 优点：速度快、比较简单；
- 缺点：得到的 attention score 可能不太准确；会导致 encoder 和 decoder 倾向于学习到相同的 embedding 空间。

additive attention 相比 multiplicative attention：
- 优点：信息更准确，encoder 和 decoder 的 embedding 空间有更多的自由度；
- 缺点：有更多超参数需要调整，参数需要更多存储空间，计算速度慢。

## Analyzing NMT Systems

### (a)

#### i

生成了两个独立的 favorite 且并不相关，原因可能是生成时并没有考虑生成句子中词语的前后联系，可以通过给输出的句子再加一层 attention 解决，这样 decoder 可以知道自己生成了什么，避免不必要的重复。

#### ii

应该使用最高级而不是比较级，可能语料库中缺少相应的最高级，可以向语料中增添相应的最高级。

#### iii

人名等专有名词没有被翻译出来，因为人名不在语料库中，可以考虑将 \<UNK> 对应的单词加入语料库中。

#### iv

西班牙语中对应的单词是多义词，对应的 apple 是英语语料库中更常见的意思，可能的解决方案是翻译成尽可能长、能表达更多含义的词组。

#### v

使用不准确，没有考虑生成句子的前后联系，可以考虑同 i 一样解决。

#### vi

不同的计量单位之间被翻译之后，对应的数值却没有被翻译，解决方案可以添加计量单位之间的换算规则，翻译计量单位之后，前面的数值做对应的转换。

### (c)

#### i

for NMT Translation $c_1$:
$$
    p_1 = \frac{1+1+1}{5} = \frac{3}{5}\\
    p_2 = \frac{1+1}{4} = \frac{1}{2}\\
    BP = 1\\
    BLEU = 1*0.3^{0.5} = 0.55
$$

for NMT Translation $c_2$:
$$
    p_1 = \frac{1+1+1+1}{5} = \frac{4}{5}\\
    p_2 = \frac{1+1}{4} = \frac{1}{2}\\
    BP = 1\\
    BLEU = 1*0.4^{0.5} = 0.63
$$

$c_2$ BLEU 成绩较高，翻译质量较好。

#### ii

for NMT Translation $c_1$:
$$
    p_1 = \frac{1+1+1}{5} = \frac{3}{5}\\
    p_2 = \frac{1+1}{4} = \frac{1}{2}\\
    BP = exp(1 - \frac{6}{5}) = 0.82\\
    BLEU = 0.82 * 0.3^{0.5} = 0.45
$$

for NMT Translation $c_2$:
$$
    p_1 = \frac{1+1}{5} = \frac{2}{5}\\
    p_2 = \frac{1}{4} = \frac{1}{4}\\
    BP = exp(1 - \frac{6}{5}) = 0.82\\
    BLEU = 0.82 * 0.2^{0.5} = 0.37
$$

$c_1$ 的 BLEU 得分较高，但其翻译质量不如 $c_2$.

#### iii

对于相同的意思可能有多种不同但都可行的翻译方式，当参考语料多的时候，能够更好地对机器给出的翻译进行评价，而降低 ii 中的问题出现的概率。参考翻译过少，可能使得质量高的翻译得分低的概率变大，无法找出真正高质量的翻译。

#### iv

优点：
- 自动化，节省人力；
- 可以适用于大规模数据；
- 同时参考很多译文，有一定的客观性。

缺点：
- 可能无法找出真正高质量的译文；
- 人工评价能够反映人对翻译质量的真实感受，所以人工评价还是评价过程中的重要部分，是不可替代的。

## 测试结果

```
bw@assignment4:~/a4$ sh ./run.sh test
[nltk_data] Downloading package punkt to /home/bw/nltk_data...
[nltk_data]   Package punkt is already up-to-date!
load test source sentences from [./en_es_data/test.es]
load test target sentences from [./en_es_data/test.en]
load model from model.bin
Decoding: 100%|███████████████████████████████████████████████████████████████████████████████████████| 8064/8064 [05:48<00:00, 23.12it/s]
Corpus BLEU: 35.79638827279159
```