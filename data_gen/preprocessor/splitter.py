#!/usr/bin/env python3
import os
import shutil

from sklearn.model_selection import train_test_split

from data_gen import io_utils

INPUT_DIR = "/home/rodneygracian/Desktop/Rod/research/projects/asl/GCN/asl_digits_recog/src/data/skeleton"
OUTPUT_DIR = "/home/rodneygracian/Desktop/Rod/research/projects/asl/GCN/asl_digits_recog/src/data/split"


def start(val_size, seed_val):
    label_path = '{}/label.json'.format(INPUT_DIR)
    print("Source directory: {}".format(INPUT_DIR))
    print("Holdout of data to '{}'...".format(OUTPUT_DIR))

    if not os.path.isfile(label_path):
        print("No data to holdout")
    else:
        # load labels for split:
        labels = io_utils.read_json(label_path)
        X = [k for k in labels]
        y = [v['label'] for (k, v) in labels.items()]

        # Holdout (train, test, val):
        X_train, X_val, y_train, y_val = holdout_data(X, y, val_size, seed_val)

        # Copy items:
        copy_items('train', 1 - val_size, X_train, INPUT_DIR, OUTPUT_DIR, labels)

        copy_items('val', val_size, X_val, INPUT_DIR, OUTPUT_DIR, labels)
        print("Holdout complete.")


def holdout_data(X, y, val_size, seed_val):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state=seed_val)
    return X_train, X_val, y_train, y_val


def copy_items(part, percent, items, input_dir, output_dir, data):
    if items:
        print("Saving '{}' data ({:.0%})...".format(part, percent))
        items_dir = '{}/{}'.format(output_dir, part)
        labels_path = '{}/{}_label.json'.format(output_dir, part)
        part_files = ['{}.json'.format(x) for x in items]
        part_labels = {x: data[x] for x in data if x in items}
        copy_files(part_files, input_dir, items_dir)
        io_utils.save_json(part_labels, labels_path)


def copy_files(items, src_dir, dest_dir):
    io_utils.ensure_dir_exists(dest_dir)

    for item in items:
        print('* {}'.format(item))
        src = '{}/{}'.format(src_dir, item)
        dest = '{}/{}'.format(dest_dir, item)
        shutil.copy(src, dest)


if __name__ == "__main__":
    val = 0.2
    seed = 2

    io_utils.remove_dir(OUTPUT_DIR)
    io_utils.create_dir(OUTPUT_DIR)

    start(val, seed)
