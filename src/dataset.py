# Load mnist dataset with torchvision
import numpy as np
from torchvision.datasets import MNIST
import torch
from torch.utils.data.dataset import Dataset


def noising_process(x, T):
    sigma = np.linspace(0.1, 0.99, T)
    noised_train_set = [x]
    for t in range(1, T):
        noised_img = (np.sqrt(sigma[-t]) * noised_train_set[t - 1] + np.sqrt(1 - sigma[-t])
                      * np.random.normal(0, 1, (28, 28)))
        noised_train_set.append(noised_img)
    return noised_train_set


def load_mnist(path: str = "../data"):
    """Load MNIST dataset."""
    train_dataset = MNIST(root=path, train=True, download=True)
    test_dataset = MNIST(root=path, train=False, download=True)
    return train_dataset, test_dataset


class Mnist_noised_dataset(Dataset):
    def __init__(self, set='train', T=100, path='../data', sigma=None):
        self.T = T
        self.sigma = np.linspace(0.1, 0.99, T)
        train = True if set == 'train' else False

        self.data = MNIST(root=path, train=train, download=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx][0]
        label = self.data[idx][1]
        data_seq = noising_process(img, self.T)
        data_seq = torch.tensor(data_seq)
        return data_seq, label

