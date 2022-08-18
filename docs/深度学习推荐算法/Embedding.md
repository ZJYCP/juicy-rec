# Embedding技术在推荐系统中

> Embedding，中文直译为嵌入，常被翻译为向量化，或者向量映射。形式上讲，Embedding是用一个低维稠密的向量表示一个对象，并且Embedding能够表达相应对象的某些特征，同时向量之间的距离可以反应对象的相似性。

## Word2Vec

由谷歌在2013年提出，是一个生成对“词”的向量表达模型。

CBOW模型的输入是$w_t$周边的词，预测的输出时$w_t$。而Skip-gram则相反，经验上讲，Skip-gram的效果更好。



**训练过程**

基于语料库生成训练样本，选取一个程度为2c+1的滑动窗口，



## Item2Vec





## Graph Embedding

 ### DeepWalk



### Node2Vec



### EGES







## 局部敏感哈希

