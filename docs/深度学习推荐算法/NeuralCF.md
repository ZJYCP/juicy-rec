# NeuralCF模型——CF与深度学习的结合

---

>NCF是协同过滤在神经网络上的实现——**神经网络协同过滤**。由新加坡国立大学与2017年提出。

## 审视矩阵分解

矩阵分解的用户隐向量和物品隐向量可以看做是embedding的结果

<img src="https://blog-1252832257.cos.ap-shanghai.myqcloud.com/20201019191102816.png" alt="img" style="zoom:67%;" />

这个模型由于结构太简单，计算也太简单， 使得输出层无法对目标进行有效的拟合。

## NCF的结构

NCF用"多层感知机+输出层"

![img](https://blog-1252832257.cos.ap-shanghai.myqcloud.com/20201019194331535.png)

使用MLP代替内积操作，一来让用户 向量和物品向量做了更多的交叉，得到更多有价值的特征组合信息，二来引入更多的非线性特征，让模型的表达能力更强。

在矩阵分解中，用户向量和物品向量进行的是内积操作，事实上，两个向量可以被任意的互操作代替，这被称为广义矩阵分解(Generalized Matrix Factorization, GMF)。

NCF中的GMF使用了元素积(点积、哈达玛积)。

![image](https://blog-1252832257.cos.ap-shanghai.myqcloud.com/690773-20210327113449474-1682294339.png)

NCF中的MF vector和MLP Vector是分别训练的，共享GMF和MLP的嵌入层可能会限制融合模型的性能。例如，它意味着，GMF和MLP必须使用的大小相同的嵌入;对于数据集，两个模型的最佳嵌入尺寸差异很大，使得这种解决方案可能无法获得最佳的组合。

## NCF的特点

NeuralCF可以看做是一种模型框架，基于用户向量和标的物embedding向量，利用不同的互操作层进行特征的交叉组合，并且连接到一起。

在实践中，要防止模型的过拟合，模型的复杂度和特征并不是越多越好。

NeuralCF模型也存在一定的局限性。它是基于协同过滤思想进行改造的，没有引入其他类型的特征。此外，在互操作的种类上也没有过多的操作，为后续的研究留下了空间。



## 代码实现

```python
# 核心代码
class SingleEmb(keras.layers.Layer):
    def __init__(self, emb_type, sparse_feature_column):
        super().__init__()
        # 取出sparse columns
        self.sparse_feature_column = sparse_feature_column
        self.embedding_layer = keras.layers.Embedding(sparse_feature_column.vocabulary_size, 
                                                      sparse_feature_column.embedding_dim, 
                                                      name=emb_type + "_" + sparse_feature_column.name)    
    
    def call(self, inputs):
        return self.embedding_layer(inputs)
    
class NearalCF(keras.models.Model):
    def __init__(self, sparse_feature_dict, MLP_layers_units):
        super().__init__()
        self.sparse_feature_dict = sparse_feature_dict
        self.MLP_layers_units = MLP_layers_units
        self.GML_emb_user = SingleEmb('GML', sparse_feature_dict['user_id'])
        self.GML_emb_item = SingleEmb('GML', sparse_feature_dict['item_id'])
        self.MLP_emb_user = SingleEmb('MLP', sparse_feature_dict['user_id'])
        self.MLP_emb_item = SingleEmb('MLP', sparse_feature_dict['item_id'])
        self.MLP_layers = []
        for units in MLP_layers_units:
            self.MLP_layers.append(keras.layers.Dense(units, activation='relu')) # input_shape=自己猜
        self.NeuMF_layer = keras.layers.Dense(1, activation='sigmoid')
        
    def call(self, X):
        #输入X为n行两列的数据，第一列为user，第二列为item
        GML_user = keras.layers.Flatten()(self.GML_emb_user(X[:,0]))
        GML_item = keras.layers.Flatten()( self.GML_emb_item(X[:,1]))
        GML_out = tf.multiply(GML_user, GML_item)
        MLP_user = keras.layers.Flatten()(self.MLP_emb_user(X[:,0]))
        MLP_item = keras.layers.Flatten()(self.MLP_emb_item(X[:,1]))
        MLP_out = tf.concat([MLP_user, MLP_item],axis=1)
        for layer in self.MLP_layers:
            MLP_out = layer(MLP_out)
        # emb的类型为int64，而dnn之后的类型为float32，否则报错
        GML_out = tf.cast(GML_out, tf.float32)
        MLP_out = tf.cast(MLP_out, tf.float32)
        concat_out = tf.concat([GML_out, MLP_out], axis=1)
        return self.NeuMF_layer(concat_out)    

```

