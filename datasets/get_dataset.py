import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def GetDataset(dataset, path):
    if dataset == 'cifar10' or dataset == 'cifar100':
        normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447],
                                         std=[0.247, 0.243, 0.262])
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        val_transform = transforms.Compose([transforms.ToTensor(), normalize])

        _dataset = datasets.CIFAR10 if dataset == 'cifar10' else datasets.CIFAR100
        train_dataset = _dataset(root=path, train=True, download=True,
                                 transform=train_transform)
        val_dataset = _dataset(root=path, train=False, download=True,
                               transform=val_transform)
    elif dataset == 'imagenet':
        traindir = os.path.join(path, 'train')
        valdir = os.path.join(path, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    return train_dataset, val_dataset
