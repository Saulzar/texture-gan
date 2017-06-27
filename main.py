from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data

import models

import tools.image.cv as cv
from tools import tensor

import dataset as d
from torch.autograd import Variable

import arguments


args = arguments.get_arguments()
print(args)

args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

def to_cuda(t):
    return t.cuda() if args.cuda else t

def var(t):
    return Variable(to_cuda(t))

try:
    os.makedirs(args.output)
except OSError:
    pass

creation_params = models.get_params(args)
model = to_cuda(models.create(creation_params))

image_size = model.image_size(args.size, args.size)

dataloader, _ = d.training(args, (image_size, image_size))
criterion = to_cuda(nn.BCELoss())

input = to_cuda(torch.FloatTensor(args.batch_size, 3, image_size, image_size))
noise = to_cuda(torch.FloatTensor(args.batch_size, args.noise_size, 1, 1))


fixed_noise = var(torch.FloatTensor(args.batch_size, args.noise_size, 1, 1).normal_(0, 1))
label = to_cuda(torch.FloatTensor(args.batch_size))
real_label = 1
fake_label = 0




# setup optimizer
optimizerD = optim.Adam(model.discriminator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
optimizerG = optim.Adam(model.generator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
# optimizerD = optim.RMSprop(model.discriminator.parameters(), lr=args.lr)
# optimizerG = optim.RMSprop(model.generator.parameters(), lr=args.lr)

#
# optimizerD  = optim.SGD(model.discriminator.parameters(), lr=args.lr, momentum=args.momentum)
# optimizerG  = optim.SGD(model.generator.parameters(), lr=args.lr, momentum=args.momentum)

def test(e):




    fake = model.generator(fixed_noise)
    tiled = tensor.tile_batch(d.un_normalize(fake.data.cpu()), cols=8)

    filename = "image{}.jpg".format(e)
    cv.imwrite(os.path.join(args.output, filename), tiled)

    latest_file = os.path.join(args.output, "latest.jpg")

    try:
        os.unlink(latest_file)
    except FileNotFoundError:
        pass
    os.symlink(filename, latest_file)

    print("written " + os.path.join(args.output, filename))
    cv.imshow("generated", tiled)


# (t[i] * log(o[i]) + (1 - t[i]) * log(1 - o[i]))

def bce(x, target):
    return -(target * x.log() + (1 - target) * (1 - x).log()).mean()

def mse(x, target):
    return (target - x).pow(2).mean()

loss = bce


for epoch in range(10000):
    for i, data in enumerate(dataloader, 0):
        #
        for p in model.parameters():
            p.data.clamp_(-1, 1)

        model.discriminator.zero_grad()
        input = var(d.normalize(data))

        real_output = model.discriminator(input)
        errD_real = loss(real_output, 1)

        # train with fake
        noise = var(model.make_noise(args.batch_size, (args.size, args.size)))
        fake = model.generator(noise)

        fake_output = model.discriminator(fake.detach())
        errD_real = loss(real_output, 0)

        errD = errD_real + errD_fake
        errD.backward()

        optimizerD.step()


        model.generator.zero_grad()
        gen_output = model.discriminator(fake)
        errG = loss(gen_output, 1)
        errG.backward()

        optimizerG.step()

        cv.waitKey(1)

        print('[%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, i, len(dataloader),
                 errD.data[0], errG.data[0], real_output.data.mean(), fake_output.data.mean(), gen_output.data.mean()))
        if i % 100 == 0:
            test(epoch)


    # do checkpointing
    # torch.save(model.generator.state_dict(), '%s/model.generator_epoch_%d.pth' % (args.output, epoch))
    # torch.save(model.discriminator.state_dict(), '%s/model.discriminator_epoch_%d.pth' % (args.output, epoch))
