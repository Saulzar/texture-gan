from tools import Struct

import torch
import torch.nn as nn
import torch.nn.functional as F


input_channels = 3

parameters = Struct(
        noise_size      = (512, "noise vector size"),
        gen_features    = (64, "hidden features in generator"),
        disc_features   = (64, "hidden features in discriminator"),
        bias = (False, "use bias in convolutions")
    )


def create(args):

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    bias = args.bias


    class Generator(nn.Module):
        def __init__(self, features):
            super(Generator, self).__init__()

            self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(     args.noise_size, features * 8, 4, 1, 0, bias=bias),
                nn.BatchNorm2d(features * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (features*8) x 4 x 4
                nn.ConvTranspose2d(features * 8, features * 4, 4, 2, 1, bias=bias),
                nn.BatchNorm2d(features * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (features*4) x 8 x 8
                nn.ConvTranspose2d(features * 4, features * 2, 4, 2, 1, bias=bias),
                nn.BatchNorm2d(features * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (features*2) x 16 x 16
                nn.ConvTranspose2d(features * 2,     features, 4, 2, 1, bias=bias),
                nn.BatchNorm2d(features),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (features) x 32 x 32
                # nn.Conv2d(    features,      features, 5, 1, 2, bias=bias),
                # nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(    features,      input_channels, 4, 2, 1, bias=bias),


                nn.Tanh()
                # state size. (nc) x 64 x 64
            )

        def forward(self, input):
            return self.main(input)



    class Discriminator(nn.Module):
        def __init__(self, features):
            super(Discriminator, self).__init__()
            self.main = nn.Sequential(
                # input is (nc) x 64 x 64
                nn.Conv2d(input_channels, features, 4, 2, 1, bias=bias),
                nn.LeakyReLU(0.2, inplace=True),

                # nn.Conv2d(    features,      features, 5, 1, 2, bias=bias),
                # nn.BatchNorm2d(features),
                # nn.LeakyReLU(0.2, inplace=True),

                # state size. (features) x 32 x 32
                nn.Conv2d(features, features * 2, 4, 2, 1, bias=bias),
                nn.BatchNorm2d(features * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (features*2) x 16 x 16
                nn.Conv2d(features * 2, features * 4, 4, 2, 1, bias=bias),
                nn.BatchNorm2d(features * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (features*4) x 8 x 8
                nn.Conv2d(features * 4, features * 8, 4, 2, 1, bias=bias),
                nn.BatchNorm2d(features * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (features*8) x 4 x 4
                nn.Conv2d(features * 8, 1, 4, 1, 0, bias=bias),
                nn.Sigmoid()
            )

        def forward(self, input):
            output = self.main(input)

            return output.view(-1, 1)



    class SGan(nn.Module):
        def __init__(self):
            super().__init__()

            self.generator = Generator(args.gen_features)
            self.discriminator = Discriminator(args.disc_features)

            self.apply(weights_init)

        def make_noise(self, batch, dim=(1, 1)):
            local_noise = torch.FloatTensor(batch, args.noise_size, dim[1], dim[0]).normal_(0, 1)
            return local_noise

        def scale_factor():
            return 16

        def forward(self, inputs):

            return self.discriminator(self.generator(inputs))

    return SGan()
