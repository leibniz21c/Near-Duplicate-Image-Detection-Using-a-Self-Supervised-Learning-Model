import torch

from torchvision import transforms
from glob import glob
from natsort import natsorted
from PIL import Image, ImageFile
from torch.utils.data import Dataset

# Set truncation config
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageDataset(Dataset):
    def __init__(self, root, transform=None):
        # Get images path
        self.base_path = root
        self.images_paths = natsorted(glob(root + "*.jpg"))
        self.images_labels = [path.split("/")[-1] for path in self.images_paths]
        self.transform = False if transform is None else transform
        
    def __getitem__(self, key):
        # index and label check
        index = self.images_paths.index(self.base_path + key) if type(key) is str else key
        if self.transform:
            return (
                self.transform(Image.open(self.images_paths[index])),
                self.images_labels[index],
            )
        return (
            Image.open(self.images_paths[index]),
            self.images_labels[index],
        )

    def __len__(self):
        return len(self.images_paths)


class PairDataset(Dataset):
    def __init__(self, root, transform=None):
        # images dataset
        self.images_dataset = ImageDataset(root=root+"images/", transform=transform)
        # loading pairs
        self.pairs = []
        with open(root + "nd_pairs.txt") as f:
            lines = f.readlines()
            for line in lines:
                i, j = line.strip().split(' ')
                self.pairs.append(((i + '.jpg', j + '.jpg'), 1))
        with open(root + "nnd_pairs.txt") as f:
            lines = f.readlines()
            for line in lines:
                i, j = line.strip().split(' ')
                self.pairs.append(((i + '.jpg', j + '.jpg'), 0))

    def __getitem__(self, index):
        return (
            self.images_dataset[self.pairs[index][0][0]],
            self.images_dataset[self.pairs[index][0][1]],
        ), self.pairs[index][1]

    def __len__(self):
        return len(self.pairs)


class SimCLRDataset(Dataset):
    def __init__(self, root, transform=None):
        # Get images path
        self.base_path = root
        self.images_paths = natsorted(glob(root + "*.jpg"))
        self.images_labels = [path.split("/")[-1] for path in self.images_paths]
        
        # transform
        self.transform = False if transform is None else transform
        
    def __getitem__(self, index):
        return (
            (
                self.transform(Image.open(self.images_paths[index])),
                self.images_labels[index],
            ), (
                self.transform(Image.open(self.images_paths[index])),
                self.images_labels[index],
            )
        )

    def __len__(self):
        return len(self.images_paths)