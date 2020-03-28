import json
import os
import shutil
import sys
import time
import traceback


def progress_bar(current, total):
    increments = 50
    percentual = ((current / total) * 100)
    i = int(percentual // (100 / increments))
    text = "\r|{0: <{1}}| {2:.0f}%".format('█' * i, increments, percentual)
    print(text, end="\n" if percentual >= 100 else "")


def ensure_dir_exists(dir):
    if not os.path.exists(dir):
        create_dir(dir)


def create_dir(dir):
    os.makedirs(dir)


def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' %
                          (class_str,
                           traceback.format_exception(*sys.exc_info())))


def remove_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir, ignore_errors=True)


def save_items(items, path):
    with open(path, 'w') as f:
        for item in items:
            f.write("{}{}".format(item, os.linesep))


def save_map(map, path):
    with open(path, 'w') as f:
        for key, val in map.items():
            f.write("{}:{}{}".format(key, val, os.linesep))


def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)


def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def create_command_line(command, args):
    command_line = command + ' '
    command_line += ' '.join(['{} {}'.format(k, v)
                              for k, v in args.items()])
    return command_line


def str2dict(v):
    return eval('dict({})'.format(v))


def print_log(str, print_time=True):
    if print_time:
        str = time.strftime("[%m.%d.%y|%X] ", time.localtime()) + str

    if print_to_screen:
        print(str)
    if save_log:
        with open('{}/log.txt'.format(work_dir), 'a') as f:
            print(str, file=f)
