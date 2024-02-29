import torchvision
import numpy as np
from torch.utils.data import Subset, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize


class Dataset:
    def __init__(self):
        transforms = Compose([
            ToTensor(),
            Normalize(mean=0, std=1)
        ])
        self.dataset = MNIST(root='data', transform=transforms, download=True)

    def get_subset(self, digit):
        return Subset(self.dataset, np.where(self.dataset.targets == digit)[0])
