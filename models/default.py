import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.model import match_size_2d
from tools import Struct

import nninit
import math

def identity(x):
    return x


#
# def selu(x):
#     alpha = 1.6732632423543772848170429916717
#     scale = 1.0507009873554804934193349852946
#
#     return scale * (F.relu(x) + alpha * (F.elu(-1*F.relu(-1*x))))

parameters = Struct(
        growth      = (2.0, "growth factor in layer features"),
        depth       = (3, "number of layers in encoder/decoder"),
        kernel_size = (5, "kernel size for convolutions"),
        dropout     = (0, "dropout per convolution"),

        discrim_bias = (1, "factor of features for discriminator vs generator"),

        nglobal     = (0, "size of global noise vector"),
        nlocal      = (32, "size of local noise vector"),
        nperiodic   = (0, "size of periodic input vector"),
        hidden      = (32, "size of features in hidden layer for estimating period")
    )

def create(args):
    class Conv(nn.Module):

        def __init__(self, in_size, out_size, kernel=args.kernel_size, dilation=1):
            super().__init__()
            self.norm = identity #nn.BatchNorm2d(out_size)
            self.conv = nn.Conv2d(in_size, out_size, kernel, dilation=dilation, padding=(kernel//2) * dilation)


        def forward(self, inputs):

            return F.relu(self.norm((self.conv(inputs))))

    # class Conv2(nn.Module):
    #
    #     def __init__(self, in_size, out_size):
    #         super().__init__()
    #         self.conv1 = Conv(in_size, out_size)
    #         self.conv2 = Conv(out_size, out_size)
    #
    #     def forward(self, inputs):
    #         return self.conv2(self.conv1(inputs))

    class Encode(nn.Module):

        def __init__(self, in_size, out_size):
            super().__init__()
            self.conv = Conv(in_size, out_size)
        #    self.down = nn.Conv2d(in_size, out_size, 3, stride=2)

        def forward(self, inputs):
            # return self.down(self.conv(inputs))

            return F.max_pool2d(self.conv(inputs), 2, 2)


    class Decode(nn.Module):

        def __init__(self, in_size, out_size):
            super().__init__()

            self.conv = Conv(in_size, in_size)
            self.up = nn.ConvTranspose2d(in_size, out_size, 2, 2)


        def forward(self, inputs):
            return self.up(self.conv(inputs))


    def features(input_channels, image_channels=3):
        def f(depth):
            assert depth <= args.depth
            if depth < args.depth - 1:
                return math.ceil(pow(1/args.growth, depth) * input_channels)
            else:
                return image_channels
        return f

    def receptive_size(depth=args.depth):
        assert depth >= 1

        filter_radius = (kernel_size+1)//2
        if(depth==1):
            return filter_radius
        else:
            return 2 * receptive_size(depth - 1) + filter_radius


    def make_grid(w, h):
        y = torch.arange(0, h).view(h, 1, 1).expand(h, w, 1)
        x = torch.arange(0, w).view(1, w, 1).expand(h, w, 1)
        return torch.cat([x, y], 2)

    def range_dim(dim, t):
        assert dim < t.dim()

        size = [1] * t.dim()
        size[dim] = t.size(dim)

        return torch.arange(0, t.size(dim)).view(size).expand_as(t)

    class Period(nn.Module):
        def __init__(self, input, output):
            super().__init__()

            self.output = output
            self.input = input

            self.conv1 = nn.Conv2d(input, args.hidden, 1)
            self.conv2 = nn.Conv2d(args.hidden, 3 * output, 1)
        def forward(self, noise):
            _, d, w, h = noise.size()
            assert d == self.input

            weight = self.conv2(self.conv1(noise))
            o = self.output
            x, y = weight.narrow(1, 0, o), weight.narrow(1, o, o)
            bias = k.narrow(1, o * 2, o)

            gx = Variable(range_dim(3, x)).type_as(noise)
            gy = Variable(range_dim(2, y)).type_as(noise)

            return torch.sin(x * gx + y * gy + bias)



    class Generator(nn.Module):

        def __init__(self, input_channels, image_channels=3):
            super().__init__()

            f = features(input_channels, image_channels)

            self.generator = nn.Sequential(
                *[Decode(f(d), f(d + 1)) for d in range(0, args.depth - 1)])


        def forward(self, noise):

            return F.tanh(self.generator(noise))

    class Discriminator(nn.Module):

        def __init__(self, input_channels, image_channels=3):
            super().__init__()

            f = features(input_channels * args.discrim_bias)

            self.discriminator = nn.Sequential(
                *[Encode(f(d + 1), f(d)) for d in reversed(range(0, args.depth - 1))])

            self.classifier = nn.Conv2d(f(0), 1, 1)

        def forward(self, inputs):
            return F.sigmoid(self.classifier(self.discriminator(inputs)))



    class SGan(nn.Module):
        def __init__(self):
            super().__init__()

            features = args.nlocal + args.nglobal
            self.generator = Generator(features)
            self.discriminator = Discriminator(features)

            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                    init.xavier_uniform(m.weight.data, gain=math.sqrt(2))
                    init.constant(m.bias.data, 0.1)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.normal_(1.0, 0.02)
                    m.bias.data.fill_(0)

        def make_noise(self, batch, dim):
            local_noise = torch.FloatTensor(batch, args.nlocal, dim[1], dim[0]).normal_(-0.5, 0.5)
            # global_noise = (torch.FloatTensor(batch, args.nglobal, 1, 1)
            #                 .normal_(0, 1).expand(batch, args.nglobal, dim[1], dim[0]))
            #
            # global_noise /= global_noise.dot(global_noise)
            #

            # return torch.cat([local_noise, global_noise], 1)
            return local_noise

        def forward(self, inputs):

            return self.discriminator(self.generator(inputs))

    return SGan()
