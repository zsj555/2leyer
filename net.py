
import numpy as np
import pickle
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

# 计算sigmoid层的反向传播导数
def sigmoid_grad(x):
        return (1.0 - sigmoid(x)) * sigmoid(x)




def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


class simpleNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        # 初始化网络
        self.params = {}
        self.params['W1'] = weight_init_std * \
                            np.random.randn(input_size, hidden_size)
                            # 用高斯分布随机初始化一个权重参数矩阵
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * \
                            np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)


    def predict(self, x):
        # 前向传播
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y


    def loss(self, x, t,reg):
        y = self.predict(x)
        loss = cross_entropy_error(y, t)+ reg*(np.sum(self.params['W2']** 2) + np.sum(self.params['W1'] ** 2))

        return loss

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum( y==t ) / float(x.shape[0])
        return accuracy



    def gradient(self, x, t,reg):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)+2 * reg * W2
        grads['b2'] = np.sum(dy, axis=0)

        dz1 = np.dot(dy, W2.T)
        da1 = sigmoid_grad(a1) * dz1
        grads['W1'] = np.dot(x.T, da1)+2 * reg * W1
        grads['b1'] = np.sum(da1, axis=0)

        return grads










import load


import matplotlib.pyplot as plt


# 导入数据
(x_train, t_train), (x_test, t_test) =load. load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []
train_acc_list = []
test_acc_list = []
test_loss_list = []

# 超参数
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1
lr_decrease=0.999995
reg=5e-6
iter_per_epoch = max(train_size / batch_size, 1)

network = simpleNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = network.gradient(x_batch, t_batch,reg)

    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch,reg)
    train_loss_list.append(loss)
    learning_rate=learning_rate*lr_decrease   # 学习率下降

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        loss = network.loss(x_test, t_test, reg)
        test_loss_list.append(loss)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

# 训练损失函数曲线
x1 = np.arange(len(train_loss_list))
plt.plot(x1, train_loss_list)
plt.xlabel("iteration")
plt.ylabel("train loss")
plt.show()


# 训练精度，测试精度曲线
markers = {'train': 'o', 'test': 's'}
x2 = np.arange(len(train_acc_list))
plt.plot(x2, train_acc_list, label='train acc')
plt.plot(x2, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()

# 测试损失函数曲线
x3 = np.arange(len(test_loss_list))
plt.plot(x3, test_loss_list)
plt.xlabel("iteration")
plt.ylabel("test loss")
plt.show()


# 保存模型
pickle.dump(network.params, open('fit.pkl', 'wb'))


# 可视化W1
W1=network.params['W1'].T
for i in range(5):
    W = W1[i].reshape(28, 28)
    plt.imshow(W)
    plt.show()




