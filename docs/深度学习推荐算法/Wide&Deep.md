
# Wide&Deep 模型

> Wide&Deep模型由谷歌于2016年年提出，被应用于Google play应用商店的APP推荐。Wide&Deep模型具备较强的记忆能力和泛化能力，不仅在当时迅速成为业界争相应用的主流模型，而且衍生出大量以Wide&Deep模型为基础的混合模型，一直影响至今。
>
> *Cheng H T, Koc L, Harmsen J, et al. Wide & deep learning for recommender systems[C]//Proceedings of the 1st workshop on deep learning for recommender systems. 2016: 7-10.*

## 模型的记忆能力和泛化能力

**记忆能力可以理解为模型直接学习并利用历史数据中的物品或者特征的”共现频率“的能力。**协同过滤、逻辑回归等简单模型具有较强的记忆能力，原始数据可以直接影响结果。模型可以产生类似于”如果点击过A，就推荐B“这类规则式的记忆，相当于模型直接记住了历史数据的分布特点，并利用这些记忆进行推荐。相反，对于多层神经网络来说，特征会被多层处理，不断与其他特征进行交叉，因此模型就一些共现的强特征的记忆反而没有简单模型深刻。

**泛化能力可以被理解为模型传递特征的相关性，以及发现稀疏甚至从未出现过的稀疏特征与最终标签相关性的能力上**。比如矩阵分解引入了隐向量，使得数据稀少的用户或者物品也能生成隐向量，从而获得有数据支撑的推荐得分，这就是非常经典的**将全局数据传递到稀疏物品上**，从而获得泛化能力的例子。深度神经网络通过特征的多次自动组合，可以深度发掘数据中潜在的模式。

在推荐系统中，记忆体现的准确性而泛化体现的是新颖性。

## Wide&Deep模型的结构

Wide&Deep 模型的直接动机就是将简单模型的记忆能力和深度神经网络的泛化能力结合。

![preview](https://blog-1252832257.cos.ap-shanghai.myqcloud.com/v2-5e7ddd792ac3f56783f6bfd3b7d32758_r.jpg)



Wide&Deep模型由单输入层的Wide部分和由Embedding层和多隐层组成的Deep部分构成。**单层的Wide部分善于处理稀疏的id特征**；Deep部分利用神经网络表达能力强的特征，**进行深层的特征交叉**，挖掘藏在特征背后的数据模式。

**Deep部分**

Google的工作中，Deep部分输入的是全量的特征向量。Deep网络是一个多层的全连接层，最后一层不进行激活
$$
a^{(l+1)} = f(W^la^l+b^l)\\
y=W^\top a^{l_f}
$$
**Wide部分**

Wide部分输入的是已安装应用和曝光应用两类id特征。这能够充分发挥Wide的记忆能力。组成这两个特征的函数被称为交叉积变换函数(cross product transformation)，定义如下：
$$
y=W^\top[X,\phi(X)]+b
$$
其等同意一个LR，$\phi(X)$表示

<img src="https://blog-1252832257.cos.ap-shanghai.myqcloud.com/v2-bfa109a472073738817ca9813d363e3d_r.jpg" alt="img" style="zoom: 67%;" />

$c_{ki}$是一个布尔变量，如果第i个特征是第k个变换的一部分则为1，反之为0.对于二值特征，一个组合特征当原特征都为0的时候才会0。

在通过交叉积变换层操作完成特征组合之后，Wide部分将组合特征输入最终的LogLoss输出层，与Deep部分的输出一同参与最后的目标拟合，完成与Deep部分的融合。

## Wide&Deep模型的进化——DCN

Deep&Cross(DCN)是2017年由斯坦福大学和谷歌的研究员提出，基本思路是由Cross网络代替了原来的Wide部分，**自动构造有限高阶的交叉特征**。

<img src="https://blog-1252832257.cos.ap-shanghai.myqcloud.com/network-structure.png" alt="img" style="zoom:50%;" />

Cross层如下：
$$
x_{l+1} = x_0x_l^\top W_l+b_l+x_l
$$
可视化

![image-20220713231437614](https://blog-1252832257.cos.ap-shanghai.myqcloud.com/image-20220713231437614.png)

交叉层在每一层均保留了输入向量，输出和输入之间的变化不会特别明显。

当cross layer叠加 $l$层时，交叉最高阶可以达到 $l+1$阶，举个栗子：

令 ![[公式]](https://www.zhihu.com/equation?tex=X_0%3D%5Cleft%5B%5Cbegin%7Bmatrix%7Dx_%7B0%2C1%7D%5C%5Cx_%7B0%2C2%7D%5Cend%7Bmatrix%7D%5Cright%5D) ，那么

![image-20220713231733413](https://blog-1252832257.cos.ap-shanghai.myqcloud.com/image-20220713231733413.png)

继续计算 ![[公式]](https://www.zhihu.com/equation?tex=X_2)，有：

![image-20220713231750316](https://blog-1252832257.cos.ap-shanghai.myqcloud.com/image-20220713231750316.png)

从公式的标红处可以看出，当cross layer叠加 $l$层时，交叉最高阶可以达到 $l+1$ 阶，并且包含了所有的交叉组合，这是DCN的精妙之处。

## Wide&Deep模型的影响力

Wide&Deep模型的影响力是巨大的，其后续的改进工作也延续至今，DeepFM、NFM都可以看做是其模型的延伸。

1. Wide&Deep模型抓住了业务问题的本质特点，能融合**传统模型的记忆能力和深度学习模型泛化能力**的优势。
2. **结构不复杂**，在工程上容易实现、训练和上线。

在此之后，越来越多的模型结构被加入到推荐系统中，深度学习模型的结构也开始朝着多样化、复杂化的方向发展。







