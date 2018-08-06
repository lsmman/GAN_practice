import torch

a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

c = torch.from_numpy(b)
print(c)

# numpy와 from_numpy로 되돌리면
# 메모리 공간을 공유한다.
a.add_(1)
print("\n\nresult : ")
print(a)
print(b)
print(c)



# tips

# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!

