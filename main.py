######################################################
# Sign Digit Recognizer built on Graph Convolutional Networks
# Author: Rodney Gracian Dsouza
######################################################

import argparse
import sys

from utils.io_utils import import_class

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parser')

    processors = dict()
    processors['preprocessing'] = import_class('data_preproc.preproc_main.Data_Preproc')
    processors['training'] = import_class('signdigit_train.SignDigit_Training')
    processors['recognition_webcam'] = import_class('signdigit_webcam.SignDigit_Webcam')
    # processors['recognition_images'] = import_class('images_gcn')

    subparsers = parser.add_subparsers(dest='processor')
    for k, p in processors.items():
        subparsers.add_parser(k, parents=[p.get_parser()])

    arg = parser.parse_args()

    Processor = processors[arg.processor]
    p = Processor(sys.argv[2:])

    p.start()
