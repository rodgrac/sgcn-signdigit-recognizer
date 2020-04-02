import os
import pickle
import sys

from numpy.lib.format import open_memmap

from data_gen.feeder_kinetics import Feeder_kinetics
from .preprocessor import Preprocessor


class Normalizer(Preprocessor):
    def __init__(self, argv=None):
        super().__init__('normalize', argv)
        self.toolbar_width = 30
        self.part = ['train', 'val']

    def start(self):
        for p in self.part:
            data_path = '{}/{}'.format(self.input_dir, p)
            label_path = '{}/{}_label.json'.format(self.input_dir, p)
            data_out_path = '{}/{}_data.npy'.format(self.output_dir, p)
            label_out_path = '{}/{}_label.pkl'.format(self.output_dir, p)

            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            self.gendata(data_path, label_path, data_out_path, label_out_path)

    def print_toolbar(self, rate, annotation=''):
        # setup toolbar
        sys.stdout.write("{}[".format(annotation))
        for i in range(self.toolbar_width):
            if i * 1.0 / self.toolbar_width > rate:
                sys.stdout.write(' ')
            else:
                sys.stdout.write('-')
            sys.stdout.flush()
        sys.stdout.write(']\r')

    def end_toolbar(self):
        sys.stdout.write("\n")

    def gendata(self, data_path, label_path, data_out_path, label_out_path):
        feeder = Feeder_kinetics(data_path=data_path, label_path=label_path)

        sample_name = feeder.sample_name
        sample_label = []

        fp = open_memmap(data_out_path, dtype='float32', mode='w+', shape=(len(sample_name), 3, 1, 21))

        for i, s in enumerate(sample_name):
            data, label = feeder[i]
            self.print_toolbar(i * 1.0 / len(sample_name),
                               '({:>5}/{:<5}) Processing data: '.format(
                                   i + 1, len(sample_name)))
            fp[i, :, :, :] = data
            sample_label.append(label)

        with open(label_out_path, 'wb') as f:
            pickle.dump((sample_name, list(sample_label)), f)
