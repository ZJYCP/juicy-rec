# Deep Crossing模型——经典的深度学习架构

---

> *Shan Y, Hoens T R, Jiao J, et al. Deep crossing: Web-scale modeling without manually crafted combinatorial features[C]//Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining. 2016: 255-262.*

## 背景

微软于2016年提出，Deep Crossing模型完整地解决了从特征工程、稀疏向量稠密化、多层神经网络进行优化目标拟合等一系列深度学习在推荐系统中的应用问题，为后续的研究奠定了良好的基础。

Deep Crossing模型的应用场景是微软搜索引擎Bing中的搜索广告推荐场景。

## 网络结构

### 特征

数值型特征counting：点击率（广告的历史点击率）、预估点击率（另一个CTR模型的CTR预估值）、广告计划中的预算

类别型特征：处理成one-hot和multi-hot的特征, 搜索广告词、广告关键词、广告标题、落地页、匹配类型等

### 结构

<img src="https://blog-1252832257.cos.ap-shanghai.myqcloud.com/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9xUDhKUm5XNlQzb0lpYnAxNEFycWlhb3ZRSjlnMUw0Z2VKbWpOU1RpYmh6TDNWaWFBZ2h4aWJPbjlUVFpiNlZ4b1F4dmQxT25DS2VUY2thbmhQalB3UkJWTmZ3LzY0MA" alt="img" style="zoom:67%;" />

Deep Crossing由四个部分组成：Embedding层、Stacking层、Multiple Residual Units层、Scoring层。



**Embedding层**：将稀疏的类别型特征转换为稠密的Embedding向量，文中使用全连接层实现。embedding层的输出向量维度远小于稀疏特征向量。其中对于连续型特征，比如feature2，不经过embedding层。

**Stacking 层**：将各个类型的特征进行拼接，也成为了concatenate层

**Multiple Residual Unit层**：多层残差网络，**对特征向量各个维度进行充分的特征交叉组合**，使得模型能够抓取更多的非线性特征和组合特征的信息

<img src="https://blog-1252832257.cos.ap-shanghai.myqcloud.com/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9xUDhKUm5XNlQzb0lpYnAxNEFycWlhb3ZRSjlnMUw0Z2VKZWdnZERLblI1a0NrejIxVGhEc3M2V2JEZTlRZFFmb2liTjFDcmtRVGljd0hQSnYzdUtOd3EwU3cvNjQw" alt="img" style="zoom:67%;" />

**Scoring层**：采用逻辑回归，对点击进行预测

## 特点与意义

从现在看，Deep Crossing是比较平淡的，没有引入注意力机制、序列模型等结构。但从历史的角度来讲，Deep Crossing结构也是具有革命性的，相比FM、FFM等只具备二阶特征交叉的模型，其能够对特征进行深度交叉。



## 代码

参见[deepcrossing](https://github.com/ZJYCP/awesome-rec/tree/main/code/DeepCrossing)
