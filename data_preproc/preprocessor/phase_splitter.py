#!/usr/bin/env python3
import os
import shutil

from sklearn.model_selection import train_test_split

from .preprocessor import Preprocessor


class Splitter(Preprocessor):
    def __init__(self, argv=None):
        super().__init__('split', argv)
        self.val = float(self.arg.split['test']) / 100.0
        self.seed = self.arg.split['seed']
        self.remove_dir('{}/split'.format(os.path.join(self.home_dir, self.work_dir)))

    def start(self):
        label_path = '{}/label.json'.format(self.input_dir)
        print("Source directory: {}".format(self.input_dir))
        print("Holdout of data to '{}'...".format(self.output_dir))

        if not os.path.isfile(label_path):
            print("No data to holdout")
        else:
            # load labels for split:
            labels = self.read_json(label_path)
            X = [k for k in labels]
            y = [v['label'] for (k, v) in labels.items()]

            # Holdout (train, test, val):
            X_train, X_val, y_train, y_val = self.holdout_data(X, y, self.val, self.seed)

            # Copy items:
            self.copy_items('train', 1 - self.val, X_train, self.input_dir, self.output_dir, labels)

            self.copy_items('val', self.val, X_val, self.input_dir, self.output_dir, labels)
            print("Holdout complete.")

    def holdout_data(self, X, y, val_size, seed_val):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, stratify=y, random_state=seed_val)
        return X_train, X_val, y_train, y_val

    def copy_items(self, part, percent, items, input_dir, output_dir, data):
        if items:
            print("Saving '{}' data ({:.0%})...".format(part, percent))
            items_dir = '{}/{}'.format(output_dir, part)
            labels_path = '{}/{}_label.json'.format(output_dir, part)
            part_files = ['{}.json'.format(x) for x in items]
            part_labels = {x: data[x] for x in data if x in items}
            self.copy_files(part_files, input_dir, items_dir)
            self.save_json(part_labels, labels_path)

    def copy_files(self, items, src_dir, dest_dir):
        self.ensure_dir_exists(dest_dir)

        for item in items:
            # print('* {}'.format(item))
            src = '{}/{}'.format(src_dir, item)
            dest = '{}/{}'.format(dest_dir, item)
            shutil.copy(src, dest)
