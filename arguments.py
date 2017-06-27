import argparse
import tools.arguments as common

import models

def get_arguments():
    parser = argparse.ArgumentParser(description='Tree segmentation')


    # Model parameters
    parser.add_argument('--nfeatures', type=int, default=8, help='number of features present in the first layer of the network')
    parser.add_argument('--input', default='/storage/workspace/dtd/images/scaly', help='input image path')
    parser.add_argument('--display', action='store_true', default=False, help='display progress of generated images')
    parser.add_argument('--image_size', type=int, default=32, help='the height / width of the input image to network')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')

    parser.add_argument('--output', default="output", help="output path")


    common.add(parser)
    models.add_arguments(parser)

    return parser.parse_args()
