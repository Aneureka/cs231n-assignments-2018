# cs231n



### 视觉处理过程

![image-20181218155216822](/Users/hiki/Library/Application Support/typora-user-images/image-20181218155216822.png)



### 一些竞赛和数据集

Dataset:

- Pascal VOC
- ImageNet
- CIFAR-10 (10 classes  50,000 training images  10,000 testing images)

2012 年，CNN 取得巨大突破（但 CNN 并不是在 2012 年被发明的, 而是来自 1998 Bell Lab.）

CNN - AlexNet - SuperVision (2012, 7 layers) ==> GoogleLeNet, Oxford VGG (19 layers) (2014) ==> MSRA Residual Network (2015, 192 layers)



### Numpy 一些注意点

##### 按列索引时两种方式的区别

```python
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

col_r1 = a[:, 1]
print(col_r1, col_r1.shape)
# Prints [ 2  6 10] (3,)

col_r2 = a[:, 1:2]
print(col_r2, col_r2.shape)
# Prints [[ 2]
#         [ 6]
#         [10]] (3, 1)
```



将 dataset 分为 training, validation, testing

交叉验证一般应用于小规模数据集，但因为极其消耗算力，因此并不常用



Multi-class SVM 在训练时，一开始的 W 一般设置为相对小（接近 0 ）的数。所以此时 s 的值接近于 0，那么损失函数就差不多是 C-1 （C 是类的数量）。所以在刚开始训练的时候，我们应该对损失函数有个预估的值，如果实际的结果与 C-1 相差很远，那么程序大概率出现了 bug 。



为了避免过拟合，通常会在损失函数中添加一个正则化项，将模型的复杂度加入到损失函数里面，以鼓励模型以某种方式选择更简单的 W （实际上使模型泛化能力更强）

$$L\left ( w \right ) = \frac{1}{N} \sum_{i=1}^{N} L_{i}\left ( f\left ( x_{i}, W \right ), y_{i} \right ) + \lambda R\left ( W \right )$$

上式的后半部分就是正则化项（Regularization）

L1 正则化倾向于使 W 矩阵变得更加稀疏，以减少一些项（特征）的影响

L2 正则化则希望 W 中的各个值都有一定作用，但作用较小，增强鲁棒性



HOG: Histogram of Oriented Gradients，方向梯度直方图

将图像按八个像素区分为八份，然后在八个像素区的每一部分，计算每个像素值的主要边缘方向，然后把这些边缘方向向量化到几个组。然后在每一个区域内计算不同的边缘方向从而得到一个直方图

![image-20181224145752329](/Users/hiki/Library/Application Support/typora-user-images/image-20181224145752329.png)



Bag of Words

ReLU 会有 dead network 的问题，learning rate 不要设置的太高，或者使用 Leaky ReLU 或 PReLU 「将 α 作为学习对象」，但效果不一定会好）

一般不会在一个 neural network 中混用不同的 activation function，虽然这并没有犯原则错误

不要使用 Sigmoid ，一般用 ReLU 就好，但要注意 learning rate 的设置，并监控 neural network 中的死节点



数据预处理：(X - mean) / std

mean（零中心化）：为了让 X 取正负，避免 gradient 都是正或都是负，从而导致学习效率低的问题

std（归一化）：使各个 feature 贡献相同（在实践中不一定会归一化，在图像处理问题中各个特征已经是相对可比较的了；但一般的机器学习问题就不一定啦）

其他预处理：PCA、Whitening



W 初始值：全0，接近0，很大或很小 都会有问题

Better practice: Xavier （ReLu会有问题）



N-layer Network: do not count the input layer

Unlike all layers in a Neural Network, the output layer neurons most commonly do not have an activation function



the regularization strength is a preferred way to control the overfitting of the neural network rather than decrease the number of the neurons and hidden layers



An important point to make about the preprocessing is that any preprocessing statistics (e.g. the data mean) must only be computed on the training data, and then applied to the validation / test data. E.g. computing the mean and subtracting it from every image across the entire dataset and then splitting the data into train/val/test splits would be a mistake. Instead, the mean must be computed only over the training data and then subtracted equally from all splits (train/val/test).



**In practice**, the current recommendation is to use ReLU units and use the `w = np.random.randn(n) * sqrt(2.0/n)`, as discussed in [He et al.](http://arxiv-web3.library.cornell.edu/abs/1502.01852).



batch normalization cannot be applied on small batch!

![image-20190102151207979](/Users/hiki/Library/Application Support/typora-user-images/image-20190102151207979.png)



![image-20190102152722865](/Users/hiki/Library/Application Support/typora-user-images/image-20190102152722865.png)

learning rate is almost the most important one among the hyperparameters

不要一次性调整超过4个hyperparameters



Sgd 产生的问题：

local minima || saddle points





