# Assignment #2

## Q.a

$$y_o = 1 \\y_w = 0$$

## Q.b

$$ \frac{\partial{J}}{\partial{v_c}} = \mathbf{U}^T(\hat{y} - y) $$

## Q.c

当 $w!=o$ 时：
$$ \frac{\partial{J}}{\partial{w}} = P(O=w|C=c)v_c $$

当 $w=o$ 时：
$$ \frac{\partial{J}}{\partial{w_o}} = (P(O=o|C=c) - 1) v_c $$

综上:
$$\frac{\partial{J}}{\partial{\mathbf{U}}} = (\hat{y} - y) v_c^T $$

## Q.d

$$\frac{\partial{\sigma(x)}}{\partial{x}} = (1 - \sigma(x))\sigma(x) $$

## Q.e

$$\frac{\partial{J}}{v_c} = -(1 - \sigma(u_o^Tv_c))u_o + \sum\limits_{k=1}^{K}(1 - \sigma(-u_k^Tv_c))u_k$$

$$\frac{\partial{J}}{u_o} = -(1 - \sigma(u_o^Tv_c))v_c$$

$$\frac{\partial{J}}{u_k} = (1 - \sigma(-u_k^Tv_c))v_c $$

不用对所有的词都计算 $exp(u_w^Tv_c) $，计算量变小。

## Q.f

$$ \frac{\partial{J}}{\partial{\mathbf{U}}} = \sum\limits_{j} \frac{\partial{J(v_c, w_{t+j}, \mathbf{U})}}{\partial{\mathbf{U}}} $$

$$ \frac{\partial{J}}{\partial{\mathbf{v_c}}} = \sum\limits_{j} \frac{\partial{J(v_c, w_{t+j}, \mathbf{U})}}{\partial{\mathbf{v_c}}} $$

$$ \frac{\partial{J}}{\partial{\mathbf{v_w}}} = 0 \  \text{when w!=c}$$