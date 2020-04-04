import argparse
import sys

from utils.io_utils import import_class

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parser')

    processors = dict()
    processors['preprocessing'] = import_class('data_gen.preproc_data.Data_Preproc')
    processors['training'] = import_class('train_gcn.SGCN_Training')
    # processors['recognition_live'] = io_utils.import_class('realtime_gcn')
    # processors['recognition_images'] = io_utils.import_class('images_gcn')

    subparsers = parser.add_subparsers(dest='processor')
    for k, p in processors.items():
        subparsers.add_parser(k, parents=[p.get_parser()])

    arg = parser.parse_args()

    Processor = processors[arg.processor]
    p = Processor(sys.argv[2:])

    p.start()
