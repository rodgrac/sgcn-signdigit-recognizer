import argparse
from distutils.util import strtobool

import yaml

from data_gen import io_utils
from data_gen.io_utils import import_class
from data_gen.io_utils import str2dict


class Data_Preproc():
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
                    io_utils.print_log('Unknown Arguments: {}'.format(k))
                    assert k in key
            parser.set_defaults(**darg)

        self.arg = parser.parse_args(argv)

    def start(self):
        dataset_dir = self.arg.dataset_dir

        if self.arg.clean_datadir:
            io_utils.remove_dir(dataset_dir)
            io_utils.create_dir(dataset_dir)

        phases = self.get_phases()

        for name, phase in phases.items():
            if name in self.arg.phases:
                self.print_phase(name)
                phase(self.arg).start()

        io_utils.print_log("Preprocessing complete")

    def get_phases(self):
        return dict(
            skeleton=import_class('preprocessor.skeleton_gen.Skeleton_Generator'),
            split=import_class('preprocessor.splitter.Splitter'),
            normalize=import_class('preprocessor.kinetics_gendata.Normalizer'),
            tfrecord=import_class('preprocessor.gen_tfrecord_data.Tfrecord_Generator')
        )

    def print_phase(self, name):
        io_utils.print_log("")
        io_utils.print_log("-" * 60)
        io_utils.print_log(name.upper())
        io_utils.print_log("-" * 60)

    @staticmethod
    def get_parser():
        parser = argparse.ArgumentParser(description="Data Preprocessing")
        parser.add_argument('-c', '--config', type=str, default=None)
        parser.add_argument('-hm', '--home_dir', type=str, default=None)
        parser.add_argument('-dd', '--dataset_dir', type=str, default=None)
        parser.add_argument('-clr', '--clean_datadir', type=strtobool, default=False)

        parser.add_argument('-ph', '--phases', type=list, default=[])
        parser.add_argument('-sk', '--skeleton', type=str2dict, default=dict())
        parser.add_argument('-fi', '--filter', type=str2dict, default=dict())
        parser.add_argument('-sp', '--split', type=str2dict, default=dict())
        parser.add_argument('-tr', '--tfrecord', type=str2dict, default=dict())
        return parser
