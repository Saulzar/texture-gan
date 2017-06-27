import torch
import torch.nn as nn
import torch.nn.functional as F


from torch.autograd import Variable

def range_dim(dim, t):
    assert dim < t.dim()

    size = [1] * t.dim()
    size[dim] = t.size(dim)

    return torch.arange(0, t.size(dim)).view(size).expand_as(t)

class Period(nn.Module):
    def __init__(self, input, output):
        super().__init__()
        hidden = 32
        self.output = output
        self.input = input

        self.conv1 = nn.Conv2d(input, hidden, 1)
        self.conv2 = nn.Conv2d(hidden, 3 * output, 1)
    def forward(self, noise):
        _, d, w, h = noise.size()
        assert d == self.input

        k = self.conv2(self.conv1(noise))
        o = self.output
        x, y, bias = k.narrow(1, 0, o), k.narrow(1,  o, o), k.narrow(1, o * 2, o)

        gx = Variable(range_dim(3, x)).type_as(noise)
        gy = Variable(range_dim(2, y)).type_as(noise)

        return torch.sin(x * gx + y * gy + bias)


p = Period(5, 7).cuda()
x = Variable(torch.arange(0, 90).view(2, 5, 3, 3).cuda())

output = p(x)
print(output.size())
