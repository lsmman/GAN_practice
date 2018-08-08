# mnist
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class Mnist_cnn(nn.Module):
    def __init__(self):
        super(Mnist_cnn, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, model):
        # print(np.shape(self.conv1(model)))
        # model = self.conv1(model)
        # print(np.shape(self.conv2(model)))
        model = self.pool1(F.relu(self.conv1(model)))
        model = self.pool2(F.relu(self.conv2(model)))
        print("사이ㅉ@ㅡ@@@@@@@@@@@@@@@@@@@@@@@@@@", np.shape(model))

        ''' view가 들어가야한다!'''
        model = model.view(model.size(0), -1)
        model = F.relu(self.fc1(model))
        model = F.relu(self.fc2(model))
        return model

net = Mnist_cnn()
print(net)

'''틀린 점 ***
    모듈을 쓸 때 대문자이고
    super(이름, self).__init__() 이라고 쳐야 super가 적용된다.
    그리고 linear 전에는 view가 들어가서 (batch size, -1) 로 펴줘야한다.
'''

input = torch.randn(1, 1, 28, 28)
out = net(input)
print("output의 사이즈는", out.size())

target = torch.tensor([3], dtype=torch.long)
loss_fn = nn.CrossEntropyLoss()  # LogSoftmax + ClassNLL Loss
err = loss_fn(out, target)
err.backward()

print(err)
print("순서대로 사이즈, norm된 weight의 값, norm화된 weight의 grad의 값")
print(net.conv1.weight.grad.size())
print(net.conv1.weight.data.norm())
print(net.conv1.weight.grad.data.norm())

