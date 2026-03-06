from typing import Dict, Tuple, Optional
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision
import torch


class MNISTDataset(Dataset):
    """
    MNIST dataset for digit classification (0-9).
    """

    LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    def __init__(
        self,
        root: str,
        split: str,  # "train" | "val"
        download: bool = True,
    ):
        self.root = root
        self.split = split

        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.1307,), (0.3081,)),
        ])

        self.label_to_index = {label: i for i, label in enumerate(self.LABELS)}
        self.index_to_label = {i: label for label, i in self.label_to_index.items()}

        if split == "val":
            self.dataset = torchvision.datasets.MNIST(root=root, train=False, download=download, transform=transform)
        else:
            self.dataset = torchvision.datasets.MNIST(root=root, train=True, download=download, transform=transform)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        x, y = self.dataset[idx]
        return x, y


def build_mnist_datasets(
    root: str,
    download: bool = True,
) -> Tuple[MNISTDataset, MNISTDataset, MNISTDataset, Dict[str, int]]:
    """
    Build MNIST datasets for train/val/test splits.

    Args:
        root: Root directory for data
        download: Whether to download the dataset if not present

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, label_mapping)
    """
    ds_train = MNISTDataset(root, "train", download=download)
    ds_val = MNISTDataset(root, "val", download=download)
    ds_test = MNISTDataset(root, "test", download=download)

    label_mapping = ds_train.label_to_index

    return ds_train, ds_val, ds_test, label_mapping
