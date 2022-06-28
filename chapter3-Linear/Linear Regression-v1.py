import torch
import random

def dataset(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w))) # 100 * 2
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

def LinearModel(X, w, b):
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def data_iter(batch_size, features, labels):
    """ yield 的作用:
        把一个函数变成一个 generator，带有 yield 的函数不再是一个普通函数，Python 解释器会将其视为一个 generator，
        调用 fab(5) 不会执行 fab 函数，而是返回一个 iterable 对象！在 for 循环执行时，每次循环都会执行 fab 函数内部的代码，
        执行到 yield b 时，fab 函数就返回一个迭代值，下次迭代时，代码从 yield b 的下一条语句继续执行，
        而函数的本地变量看起来和上次中断执行前是完全一样的，于是函数继续执行，直到再次遇到 yield。
    """
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)]
        )
        yield features[batch_indices], labels[batch_indices]

def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

def train():
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = dataset(true_w, true_b, 1000) # y = w1 * x1 + w2 * x2 + b
    model = LinearModel
    loss = squared_loss
    opim = sgd
    w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)    
    lr = 0.03
    num_epochs = 3
    batch_size = 10
    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            ls = loss(model(X, w, b), y) # X和y的小批量损失 （100）
            # 因为l形状是(batch_size, 1)，而不是一个标量。l中的所有元素被加到一起，
            # 并以此计算关于[w,b]的梯度
            ls.sum().backward()
            opim([w, b], lr, batch_size)
        with torch.no_grad():
            train_ls = loss(model(features, w, b), labels)
            print(f'epoch{epoch + 1}, loss {float(train_ls.mean()):f}')
    print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
    print(f'b的估计误差：{true_b - b}')


if __name__ == '__main__':
    train()