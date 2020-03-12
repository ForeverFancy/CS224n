# SQuAD

## 过程

### model

注意：Linear 的 feature 映射为 hidden_size -> 2。

这里其实要分开 (torch.split)，分别表示句子中的单词作为 answer 中 start_token 的成绩和作为 end_token 的成绩。然后分别用这两个成绩计算 loss。