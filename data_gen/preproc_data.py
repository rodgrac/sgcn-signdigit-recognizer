import argparse
from distutils.util import strtobool

import yaml

from .io_utils import IO
from .io_utils import import_class
from .io_utils import str2dict
from .io_utils import str2list


class Data_Preproc(IO):
    def __init__(self, argv=None):
        self.load_arg(argv)
        super().__init__(self.arg)

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

        if self.arg.clean_homedir:
            self.remove_dir(self.home_dir)
            self.create_dir(self.home_dir)

        phases = self.get_phases()

        print(self.arg.phases)

        for name, phase in phases.items():
            if name in self.arg.phases:
                self.print_phase(name)
                phase(self.arg).start()

        self.print_log("Preprocessing complete")

    def get_phases(self):
        return dict(
            skeleton=import_class('data_gen.preprocessor.skeleton_gen.Skeleton_Generator'),
            split=import_class('data_gen.preprocessor.splitter.Splitter'),
            normalize=import_class('data_gen.preprocessor.kinetics_gendata.Normalizer'),
            tfrecord=import_class('data_gen.preprocessor.gen_tfrecord_data.Tfrecord_Generator')
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
        parser.add_argument('-clr', '--clean_homedir', type=strtobool, default=False)

        parser.add_argument('--save_log', type=strtobool, default=True)
        parser.add_argument('--print_log', type=strtobool, default=True)

        parser.add_argument('-ph', '--phases', type=str2list, default=[])
        parser.add_argument('-sk', '--skeleton', type=str2dict, default=dict())
        parser.add_argument('-nm', '--normalize', type=str2dict, default=dict())
        parser.add_argument('-sp', '--split', type=str2dict, default=dict())
        parser.add_argument('-tr', '--tfrecord', type=str2dict, default=dict())
        return parser
