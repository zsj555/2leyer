# 训练和测试步骤
## 下载和读取MINIST数据集
在网上下载MINIST数据集，并将压缩文件和代码放在同一目录下，通过load.py读取数据集
## 设计激活函数
采用sigmoid函数作为激活函数，并且使用归一化指数函数softmax
```
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))
```
## 采用交叉熵计算损失函数
```
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
```
## 定义两层神经网络类
### 1、定义参数W1、W2、b1、b2
### 2、前向传播
### 3、计算损失函数
### 4、计算正确率
### 5、反向传播计算梯度
## 开始训练
调试超参数。经测试，步长iters_num = 10000  ，batch_size = 100，学习率learning_rate = 0.1，学习率下降速度lr_decrease=0.999995，正则项reg=5e-6，中间层大小为50。
训练模型并测试。
给出随步长增长，训练和测试的准确率。
## 画图与可视化
画出训练和测试的loss曲线、accuracy曲线。
选取W1中前5维，转化为28*28的矩阵，进行可视化。
## 保存模型

