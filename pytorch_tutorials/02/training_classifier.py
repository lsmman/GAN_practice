

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


def run():
    torch.multiprocessing.freeze_support()
    print('loop')
    #
    # # 1. Loading and normalizing CIFAR10
    #
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                            download=True, transform=transforms)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transforms)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=0)
    print('done')

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def imshow(img):
        img = img / 2 + 0.5
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    # images, labels = iter(trainloader).next()
    images, labels = trainset[0]

    imshow(torchvision.utils.make_grid(images))
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


if __name__ == '__main__':
    run()
