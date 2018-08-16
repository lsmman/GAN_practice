import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

path = '../../data'
batch_size = 100

train_dataset = torchvision.datasets.MNIST(root=path, train=True,transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root=path, train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

#
#     print(image, label)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.pool1 = nn.MaxPool2d(kernel_size=5)
        self.pool2= nn.MaxPool2d(kernel_size=5)

        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x)))
        x = self.conv2_drop(self.conv2(x))
        x = F.relu(self.pool2(x))
        x = x.view(-1, 320)

net = Net()
for image, label in test_loader):
    image = image.reshape(-1, 28*28)
    output = net(image)
    print(output)
