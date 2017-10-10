import argparse
import os

import bson
import pandas as pd
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description='convert BSON to files')
    parser.add_argument('--root', '-r', default='data/')
    parser.add_argument('--data', '-d',
                        choices=['train', 'test', 'train_example'],
                        required=True)
    parser.add_argument('--not_display_progress_bar', action='store_true')
    args = parser.parse_args()

    train_flag = True if args.data in ['train', 'train_example'] else False
    if args.data == 'train_example':
        bar_flag = False
    else:
        bar_flag = not args.not_display_progress_bar

    out_folder = os.path.join(args.root, args.data)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    if train_flag:
        categories = pd.read_csv(os.path.join(
            args.root, 'category_names.csv'), index_col='category_id')

        for category in categories.index:
            sub_folder = os.path.join(out_folder, str(category))
            if not os.path.exists(sub_folder):
                os.mkdir(sub_folder)

    num_products = 7069896 if train_flag else 1768182
    if bar_flag:
        bar = tqdm(total=num_products)
    bson_file = os.path.join(args.root, '{}.bson'.format(args.data))
    with open(bson_file, 'rb') as fbson:
        for c, d in enumerate(bson.decode_file_iter(fbson)):
            if train_flag:
                category = d['category_id']
            _id = d['_id']
            for e, pic in enumerate(d['imgs']):
                if train_flag:
                    fname = os.path.join(
                        out_folder, str(category), '{}-{}.jpg'.format(_id, e))
                else:
                    fname = os.path.join(
                        out_folder, '{}-{}.jpg'.format(_id, e))
                with open(fname, 'wb') as f:
                    f.write(pic['picture'])
            if bar_flag:
                bar.update()


if __name__ == '__main__':
    main()
