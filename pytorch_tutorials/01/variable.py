import torch
import torchvision

x = torch.empty(5,3)
print("empty : \n",x)

x = torch.rand(5,3)
print("rand : \n",x)

x = torch.zeros(5,3, dtype=torch.long)
print("zeros : \n",x)

x = torch.tensor([5.5,3])
print("tensor: \n", x)

x = x.new_ones(5, 3, dtype=torch.double)
print("new_ones", x)

#like는 전의 특성을 그대로 가져간다.
x = torch.randn_like(x, dtype=torch.float)
print("randn_like", x)

print("size : ", x.size())

y = torch.zeros(5, 3)
result = x+y
print("x+y : ", result)

result  = torch.empty(5,3)
torch.add(x, y, out=result)
print("add_function(x+y) : ", result)

print("before adding\n",y)
y.add_(x)
print("after adding\n", y)

print(y[:, 0])
print(y[0, :])

x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())

x = torch.rand(1)
print(x.item())

