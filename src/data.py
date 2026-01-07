# data.py

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import os
import matplotlib.pyplot as plt
import math

def get_cifar10_loaders(batch_size=32, data_dir=r"data", sanity_check=False, show_only_transf=False):
    
    os.makedirs(data_dir,exist_ok=True)
    transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),std=(0.2023, 0.1994, 0.2010))
    ])
    transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),std=(0.2023, 0.1994, 0.2010))
    ])
    trainset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform_train
    )

    testset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform_test
    )
    # SANITY CHECK: CAN THE MODEL MEMORIZE?
    if(sanity_check):
        indextrain = [i for i in range(0,500)]
        indextest = [i for i in range(0,100)]
        
        trainset = Subset(trainset,indextrain)
        testset = Subset(testset, indextest)

    # SHOW ONLY TRANSFORMATIONS OF TRAINSET AND END EXECUTION
    if(show_only_transf):
        show_versions(trainset)
        show_batch(x=[trainset[i][0] for i in range(0,batch_size)])
        raise RuntimeError("ShowOnlyTransf")

    nworkers = 10
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=nworkers, persistent_workers=True, pin_memory=True)
    testloader  = DataLoader(testset,  batch_size=batch_size, shuffle=False, num_workers=nworkers, persistent_workers=True, pin_memory=True)

    example_img, _ = trainset[0]

    return trainloader, testloader, example_img


def show_versions(dataset, idx=28, n=12, rows=3, cols=4, title=None):
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    if title: fig.suptitle(title)

    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.axis("off"); continue
        img, _ = dataset[idx]      # Each acess -> different augmentations
        ax.imshow(img.permute(1,2,0))
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def show_batch(x, batch_size=32, title=None):
    rows = math.floor(math.sqrt(batch_size))
    cols = math.ceil(batch_size/rows)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*1.5, rows*1.5))
    if title: fig.suptitle(title)

    for i, ax in enumerate(axes.flat):
        if i >= batch_size:
            ax.axis("off"); continue
        img = x[i].permute(1,2,0)
        ax.imshow(img)
        ax.axis("off")

    plt.tight_layout()
    plt.show()