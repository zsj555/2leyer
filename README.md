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
