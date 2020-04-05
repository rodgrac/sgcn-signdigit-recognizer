import argparse
import os
from distutils.util import strtobool

import yaml

from utils.io_utils import IO
from utils.io_utils import import_class
from utils.io_utils import str2dict
from utils.io_utils import str2list


class Data_Preproc(IO):
    def __init__(self, argv=None):
        self.load_arg(argv)
        super().__init__(self.arg)
        self.save_arg(self.arg, 'preproc')
        self.work_dir = self.arg.work_dir

    def load_arg(self, argv=None):
        parser = self.get_parser()
        p = parser.parse_args(argv)

        if p.config:
            with open(p.config, 'r') as f:
                darg = yaml.load(f)

            key = vars(p).keys()
            for k in darg.keys():
                if k not in key:
                    print('Unknown Arguments: {}'.format(k))
                    assert k in key
            parser.set_defaults(**darg)

        self.arg = parser.parse_args(argv)

    def start(self):

        if self.arg.clean_workdir:
            self.remove_dir(os.path.join(self.home_dir, self.work_dir))
            self.create_dir(os.path.join(self.home_dir, self.work_dir))

        phases = self.get_phases()

        print(self.arg.phases)

        for name, phase in phases.items():
            if name in self.arg.phases:
                self.print_phase(name)
                phase(self.arg).start()

        self.print_log("Preprocessing complete")

    def get_phases(self):
        return dict(
            skeleton=import_class('data_preproc.preprocessor.phase_skeleton.Skeleton_Generator'),
            split=import_class('data_preproc.preprocessor.phase_splitter.Splitter'),
            normalize=import_class('data_preproc.preprocessor.phase_normalize.Normalizer'),
            tfrecord=import_class('data_preproc.preprocessor.phase_tfrecord.Tfrecord_Generator')
        )

    def print_phase(self, name):
        self.print_log("")
        self.print_log("-" * 60)
        self.print_log(name.upper())
        self.print_log("-" * 60)

    @staticmethod
    def get_parser(add_help=False):
        parser = argparse.ArgumentParser(add_help=add_help, description="Data Preprocessing")
        parser.add_argument('-c', '--config', type=str, default=None)
        parser.add_argument('-dd', '--home_dir', type=str, default=None)
        parser.add_argument('-ds', '--dataset_dir', type=str, default=None)
        parser.add_argument('-clr', '--clean_workdir', type=strtobool, default=False)

        parser.add_argument('--work_dir', type=str, default='data')
        parser.add_argument('--save_log', type=strtobool, default=True)
        parser.add_argument('--print_log', type=strtobool, default=True)
        parser.add_argument('--log_dir', type=str, default='logs')

        parser.add_argument('-ph', '--phases', type=str2list, default=[])
        parser.add_argument('-sk', '--skeleton', type=str2dict, default=dict())
        parser.add_argument('-nm', '--normalize', type=str2dict, default=dict())
        parser.add_argument('-sp', '--split', type=str2dict, default=dict())
        parser.add_argument('-tr', '--tfrecord', type=str2dict, default=dict())

        return parser
