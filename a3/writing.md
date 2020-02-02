# Assignment 3

## 1

### a.i

倾向于保持原来的改变量，新计算出来的梯度占比较小，作微调。对于未改变方向的梯度方向，步长变长；反之相反，这样可以加快收敛。

### a.ii

学习率变化，动态调整学习率，避免步长过大或过小。

### b.i

$$\gamma = \frac{1}{1-p_{drop}}$$

### b.ii

这是正则化防止过拟合的一种方式，在训练阶段使用，而如果在预测阶段使用可能会丢弃掉重要的节点因而实际预测产生误差。

## 2

### a

| Stack                          | Buffer                      | New dependency      | Transition |
| ------------------------------ | --------------------------- | ------------------- | ---------- |
| [ROOT, parsed]                 | [this, sentence, correctly] | parsed -> I         | LEFT-ARC   |
| [ROOT, parsed, this]           | [sentence, correctly]       |                     | SHIFT      |
| [ROOT, parsed, this, sentence] | [correctly]                 |                     | SHIFT      |
| [ROOT, parsed, sentence]       | [correctly]                 | this -> sentence    | LEFT-ARC   |
| [ROOT, parsed]                 | [correctly]                 | sentence -> parsed  | RIGHT-ARC  |
| [ROOT, parsed, correctly]      | []                          |                     | SHIFT      |
| [ROOT, parsed]                 | []                          | correctly -> parsed | RIGHT-ARC  |
| [ROOT]                         | []                          | ROOT -> parsed      | RIGHT-ARC  |

### b

如果计算初始化则共需要 1 + n + n = 2n + 1 步，不计算初始化则需要 2n 步。