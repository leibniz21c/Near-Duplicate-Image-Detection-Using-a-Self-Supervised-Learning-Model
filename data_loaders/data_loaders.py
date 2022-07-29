from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import SimCLRDataset, PairDataset


class SimCLRDataLoader(DataLoader):
    def __init__(self, root, train=True, batch_size=8, shuffle=False, num_workers=0):

        #################################################
        #                   Transforms                  #
        #################################################
        color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        if train:
            trsfm = transforms.Compose([
                transforms.Resize(336),
                transforms.RandomResizedCrop((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
            ])
        else:
            trsfm = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        #################################################

        self.dataset = SimCLRDataset(root=root + "images/", transform=trsfm)
        super().__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )


class PairDataLoader(DataLoader):
    def __init__(self, root, batch_size=8, shuffle=False, num_workers=0):
        #################################################
        #                   Transforms                  #
        #################################################
        trsfm = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
        ])
        #################################################
        self.dataset = PairDataset(
            root=root, 
            transform=trsfm
        )
        super().__init__(
            self.dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers
        )