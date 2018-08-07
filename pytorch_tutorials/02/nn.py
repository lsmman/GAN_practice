
### A typical training procedure for a neural network is as follows:
# Define the neural network that has some learnable parameters (or weights)
# Iterate over a dataset of inputs
# Process input through the network
# Compute the loss (how far is the output from being correct)
# Propagate gradients back into the network’s parameters
# Update the weights of the network, typically using a simple update rule: weight = weight - learning_rate * gradient

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1,6,5)
        self.conv2 = nn.Conv2d(6,16,5)

        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84,10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)

params = list(net.parameters())
print(len(params))
print(params[0].size())

input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

net.zero_grad()
# out.backward(torch.rand(1, 10))

###### LOSS function

output = net(input)
target = torch.randn(10)
target = target.view(1, -1)
criterion = nn.MSELoss()
loss = criterion(out, target)
print(loss)
print(loss.grad_fn)
print(loss.grad_fn.next_functions[0][0])
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])

##### Backprop

net.zero_grad()

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)
net.conv1.bias.requires_grad_(True)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

##### update the weights

# weight = weight- learning_rate * gradient

learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)

optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

optimizer.zero_grad()
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()

print('conv1.bias.grad after backward after optimizer')
print(net.conv1.bias.grad)