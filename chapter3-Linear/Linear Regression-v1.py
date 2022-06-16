import torch
import random

def dataset(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

def LinearModel(X, w, b):
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def data_iter(batch_size, features, labels):
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
    features, labels = dataset(true_w, true_b, 1000)
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
            ls = loss(model(X, w, b), y)
            ls.sum().backward()
            opim([w, b], lr, batch_size)
        with torch.no_grad():
            train_ls = loss(model(features, w, b), labels)
            print(f'epoch{epoch + 1}, loss {float(train_ls.mean()):f}')
    print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
    print(f'b的估计误差：{true_b - b}')


if __name__ == '__main__':
    train()