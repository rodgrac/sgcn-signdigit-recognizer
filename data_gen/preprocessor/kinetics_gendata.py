import os
import sys
import pickle
import argparse

import numpy as np
from numpy.lib.format import open_memmap

from data_gen.feeder_kinetics import Feeder_kinetics

INPUT_DIR = "/home/rodneygracian/Desktop/Rod/research/projects/asl/GCN/asl_digits_recog/src/data/split"
OUTPUT_DIR = "/home/rodneygracian/Desktop/Rod/research/projects/asl/GCN/asl_digits_recog/src/data/normalized"

toolbar_width = 30


def print_toolbar(rate, annotation=''):
    # setup toolbar
    sys.stdout.write("{}[".format(annotation))
    for i in range(toolbar_width):
        if i * 1.0 / toolbar_width > rate:
            sys.stdout.write(' ')
        else:
            sys.stdout.write('-')
        sys.stdout.flush()
    sys.stdout.write(']\r')


def end_toolbar():
    sys.stdout.write("\n")


def gendata(data_path, label_path, data_out_path, label_out_path):
    feeder = Feeder_kinetics(data_path=data_path, label_path=label_path)

    sample_name = feeder.sample_name
    sample_label = []

    fp = open_memmap(data_out_path, dtype='float32', mode='w+', shape=(len(sample_name), 3, 1, 21))

    for i, s in enumerate(sample_name):
        data, label = feeder[i]
        print_toolbar(i * 1.0 / len(sample_name),
                      '({:>5}/{:<5}) Processing data: '.format(
                          i + 1, len(sample_name)))
        fp[i, :, :, :] = data
        sample_label.append(label)

    with open(label_out_path, 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)


if __name__ == '__main__':

    part = ['train', 'val']
    for p in part:
        data_path = '{}/{}'.format(INPUT_DIR, p)
        label_path = '{}/{}_label.json'.format(INPUT_DIR, p)
        data_out_path = '{}/{}_data.npy'.format(OUTPUT_DIR, p)
        label_out_path = '{}/{}_label.pkl'.format(OUTPUT_DIR, p)

        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        gendata(data_path, label_path, data_out_path, label_out_path)
