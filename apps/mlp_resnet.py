import sys
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)

def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    module = nn.Sequential(
        # there should be bias=False, since we have normalization after that
        nn.Linear(dim, hidden_dim, bias=True),
        norm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(drop_prob),
        nn.Linear(hidden_dim, dim, bias=True),
        norm(dim),
    )
    return nn.Sequential(
        nn.Residual(module),
        nn.ReLU(),
    )


def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    return nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.ReLU(),
        *[ResidualBlock(hidden_dim, hidden_dim // 2, norm=norm, drop_prob=drop_prob) for _ in range(num_blocks)],
        nn.Linear(hidden_dim, num_classes),
    )




def epoch(dataloader, model, opt=None):
    np.random.seed(4)

    model.train()
    if opt is None:
        model.eval()
    
    loss_func = nn.SoftmaxLoss()

    loss_sum = accuracy = count = 0

    for batch in dataloader:
        features, labels = batch
        features = features.reshape((features.shape[0], np.prod(features.shape[1:])))

        if opt is not None:
            opt.reset_grad()

        logits = model(features)
        loss = loss_func(logits, labels)

        if opt is not None:
            loss.backward()
            opt.step()

        count += features.shape[0]
        loss_sum += loss.detach().numpy() * features.shape[0]
        accuracy += (logits.detach().numpy().argmax(-1) == labels.detach().numpy()).sum()
        
    avg_loss = loss_sum / count
    avg_error_rate = 1 - accuracy / count

    return avg_error_rate, avg_loss



def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    
    mnist_train_dataset = ndl.data.MNISTDataset(
        f"{data_dir}/train-images-idx3-ubyte.gz",
        f"{data_dir}/train-labels-idx1-ubyte.gz",
        # transforms=[ndl.data.RandomFlipHorizontal(), ndl.data.RandomCrop()],
        transforms=None,
    )
    mnist_test_dataset = ndl.data.MNISTDataset(
        f"{data_dir}/t10k-images-idx3-ubyte.gz",
        f"{data_dir}/t10k-labels-idx1-ubyte.gz",
        transforms=None,
    )

    mnist_train_dataloader = ndl.data.DataLoader(
        dataset=mnist_train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    mnist_test_dataloader = ndl.data.DataLoader(
        dataset=mnist_test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    model = MLPResNet(dim=28 * 28, hidden_dim=hidden_dim)

    optim = None
    if optimizer is not None:
        optim = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_error = train_loss = test_error = test_loss = 0
    for _ in range(epochs):
        train_error, train_loss = epoch(mnist_train_dataloader, model, optim)

    test_error, test_loss = epoch(mnist_test_dataloader, model, None)

    return train_error, train_loss, test_error, test_loss
        


if __name__ == "__main__":
    train_mnist(data_dir="../data")
