import os

from data_gen import io_utils


class Preprocessor():
    def __init__(self, phase_name, argv=None):
        super().__init__(argv)
        self.phase_name = phase_name
        self.input_dir, self.output_dir = self.get_input_output_dir(phase_name)
        io_utils.ensure_dir_exists(self.output_dir)

    def start(self):
        pass

    def get_input_output_dir(self, phase_name):
        input_dir = None
        output_dir = None
        work_dir = self.arg.home_dir
        phase_args = self.arg.__getattribute__(phase_name)

        if 'input_dir' in phase_args:
            str_input_dir = '{}/{}'.format(work_dir, phase_args['input_dir'])
            input_dir = os.path.realpath(str_input_dir)
        if 'output_dir' in phase_args:
            str_output_dir = '{}/{}'.format(work_dir, phase_args['output_dir'])
            output_dir = os.path.realpath(str_output_dir)
        return input_dir, output_dir