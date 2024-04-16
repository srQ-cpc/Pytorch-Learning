import torch

x = torch.arange(12)
print(x)
print(x.shape)
print(x.numel())

x = x.reshape(3, 4)
print(x)

print(torch.zeros((2, 3, 4)))
print(torch.ones((2, 3, 4)))
print(torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]]))

x = torch.tensor([1.0, 2, 3, 4])  # 特意创建了float
y = torch.tensor(([2, 2, 2, 2]))

print(x + y)
print(x - y)
print(x * y)
print(x / y)
print(x ** y)
print(torch.exp(x))

x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
y = torch.arange(12, dtype=torch.float32).reshape(3, 4) + 10

print(torch.cat((x, y), dim=0))
print(torch.cat((x, y), dim=1))
print(x == y)
print(x.sum())


# 广播
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
print(a + b)

print(x[-1])
print(x[1: 3])

x[1, 2] = 9
print(x)

x[0: 2, :] = 12
print(x)

#内存
before = id(y)
y = y + x
print(id(y) == before)


before = id(y)
y += x  # y[:] = x + y 等价
print(id(y) == before)


a = x.numpy()
b = torch.tensor(a)
print(type(a), type(b))


a = torch.tensor([3.5])
print(a, a.item(), float(a), int(a))