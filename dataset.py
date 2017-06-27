import os
import argparse
from torch.utils.data import DataLoader

from tools.image import cv

from tools.dataset.flat import FlatFolder, Preloaded, image_file
from tools.dataset.samplers import RepeatSampler
from tools import tensor, Struct

import random


def random_check(lower, upper):
    if (lower >= upper):
        return (lower + upper) / 2
    else:
        return random.randint(lower, upper)

def random_region(image, size, border = 0):

    w, h = image.size(1), image.size(0)
    tw, th = size

    x1 = random_check(border, w - tw - border)
    y1 = random_check(border, h - th - border)

    return (x1, y1), (x1 + tw, y1 + th)

def random_crop(dim, border=0):
    def crop(image):
        h, w, c = image.size()
        assert dim[0] + border <= w and dim[1] + border <= h

        pos, _ = random_region(image, dim, border)
        return image.narrow(0, pos[1], dim[1]).narrow(1, pos[0], dim[0])
    return crop


def load_rgb(filename):
    return cv.imread(filename)

default_statistics = Struct(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

def normalize(image, mean=default_statistics.mean, std=default_statistics.mean):
    image = image.float().div_(255)
    for i in range(0, 2):
        image.select(2, i).sub_(mean[i]).div_(std[i])
    return image.permute(0, 3, 1, 2)


def un_normalize(image, mean=default_statistics.mean, std=default_statistics.mean):
    image = image.clone()

    for i in range(0, 2):
        image.select(2, i).mul_(std[i]).add_(mean[i])

    image = image.mul_(255).clamp(0, 255).byte()
    return image.permute(0, 2, 3, 1)



def training(args, dim):

    assert os.path.isdir(args.input) or os.path.isfile(args.input)
    dataset = None

    if os.path.isfile(args.input):
        data = load_rgb(args.input)
        dataset = Preloaded(args.input, data=[data], transform=random_crop(dim))
    else:
        dataset = FlatFolder(args.input,
            file_filter=image_file,
            loader=load_rgb,
            transform=random_crop(dim))

    loader=DataLoader(dataset, num_workers=args.num_workers,
        batch_size=args.batch_size, sampler=RepeatSampler(args.epoch_size, dataset))

    return loader, dataset




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Segmentation - view data set')
    parser.add_argument('--input', default='/storage/workspace/dtd/images/scaly',
                        help='input image path')


    args = parser.parse_args()

    args.num_workers = 1
    args.epoch_size = 1024
    args.batch_size = 16

    loader, dataset = training(args, (144, 144))

    for _, data in enumerate(loader):
        image = tensor.tile_batch(data, cols=4)

        if (cv.display(image) == 27):
            break
