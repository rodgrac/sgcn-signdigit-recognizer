import json
import os
import re
import shutil
import sys
import time
import traceback

import yaml


class IO:

    def __init__(self, argv=None):
        self.arg = argv
        self.home_dir = self.arg.home_dir
        self.log_dir = self.arg.log_dir
        self.save_log = self.arg.save_log
        self.print_to_screen = self.arg.print_log
        self.cur_time = time.time()

    def progress_bar(self, current, total):
        increments = 50
        percentual = ((current / total) * 100)
        i = int(percentual // (100 / increments))
        text = "\r|{0: <{1}}| {2:.0f}%".format('█' * i, increments, percentual)
        print(text, end="\n" if percentual >= 100 else "")

    def ensure_dir_exists(self, dir):
        if not os.path.exists(dir):
            self.create_dir(dir)

    def create_dir(self, dir):
        os.makedirs(dir)

    def remove_dir(self, dir):
        if os.path.exists(dir):
            shutil.rmtree(dir, ignore_errors=True)

    def save_items(self, items, path):
        with open(path, 'w') as f:
            for item in items:
                f.write("{}{}".format(item, os.linesep))

    def save_map(self, map, path):
        with open(path, 'w') as f:
            for key, val in map.items():
                f.write("{}:{}{}".format(key, val, os.linesep))

    def save_json(self, data, path):
        with open(path, 'w') as f:
            json.dump(data, f)

    def read_json(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def create_command_line(self, command, args):
        command_line = command + ' '
        command_line += ' '.join(['{} {}'.format(k, v)
                                  for k, v in args.items()])
        return command_line

    def print_log(self, log, print_time=True):
        if print_time:
            log = time.strftime("[%m.%d.%y|%X] ", time.localtime()) + log

        if self.print_to_screen:
            print(log)
        if self.save_log:
            with open('{}/{}/log.txt'.format(self.home_dir, self.log_dir), 'a') as f:
                print(log, file=f)

    def save_arg(self, arg, p_type):
        if p_type == 'preproc':
            session_file = '{}/{}/preproc_config.yaml'.format(self.home_dir, self.log_dir)
        elif p_type == 'training':
            session_file = '{}/{}/train_config.yaml'.format(self.home_dir, self.log_dir)
        elif p_type == 'testing':
            session_file = '{}/{}/test_config.yaml'.format(self.home_dir, self.log_dir)

        # save arg
        arg_dict = vars(arg)
        if not os.path.exists(os.path.join(self.home_dir, self.log_dir)):
            os.makedirs(os.path.join(self.home_dir, self.log_dir))
        with open(session_file, 'w') as f:
            f.write('# command line: {}\n\n'.format(' '.join(sys.argv)))
            yaml.dump(arg_dict, f, default_flow_style=False, indent=4)


def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' %
                          (class_str,
                           traceback.format_exception(*sys.exc_info())))


def str2dict(v):
    return eval('dict({})'.format(v))


def str2list(strlist):
    if strlist is not None:
        strlist = strlist.strip(', ')
    if not strlist:
        return []
    return re.split("[, ]+", strlist)
