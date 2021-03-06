import os

from utils.io_utils import IO


class Preprocessor(IO):
    def __init__(self, phase_name, argv=None):
        super().__init__(argv)
        self.phase_name = phase_name
        self.work_dir = self.arg.work_dir
        self.dataset_dir = self.arg.dataset_dir
        self.input_dir, self.output_dir = self.get_input_output_dir(phase_name)
        self.labels_list = self.arg.labels
        self.ensure_dir_exists(self.output_dir)

    def start(self):
        pass

    def get_input_output_dir(self, phase_name):
        input_dir = None
        output_dir = None
        tmp_dir = os.path.join(self.home_dir, self.work_dir)
        phase_args = self.arg.__getattribute__(phase_name)

        if 'input_dir' in phase_args:
            str_input_dir = '{}/{}'.format(tmp_dir, phase_args['input_dir'])
            input_dir = os.path.realpath(str_input_dir)
        if 'output_dir' in phase_args:
            str_output_dir = '{}/{}'.format(tmp_dir, phase_args['output_dir'])
            output_dir = os.path.realpath(str_output_dir)
        return input_dir, output_dir
