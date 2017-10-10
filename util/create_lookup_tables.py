import argparse
from collections import defaultdict
import os
import struct

import bson
import numpy as np
import pandas as pd
from tqdm import tqdm


def make_category_tables(categories_df):
    cat2idx = {}
    idx2cat = {}
    for ir in categories_df.itertuples():
        category_id = ir[0]
        category_idx = ir[4]
        cat2idx[category_id] = category_idx
        idx2cat[category_idx] = category_id
    return cat2idx, idx2cat


def read_bson(bson_path, num_records, with_categories):
    rows = {}
    with open(bson_path, 'rb') as f, tqdm(total=num_records) as pbar:
        offset = 0
        while True:
            item_length_bytes = f.read(4)
            if len(item_length_bytes) == 0:
                break

            length = struct.unpack('<i', item_length_bytes)[0]

            f.seek(offset)
            item_data = f.read(length)
            assert len(item_data) == length

            item = bson.BSON.decode(item_data)
            product_id = item['_id']
            num_imgs = len(item['imgs'])

            row = [num_imgs, offset, length]
            if with_categories:
                row += [item['category_id']]
            rows[product_id] = row

            offset += length
            f.seek(offset)
            pbar.update()

    columns = ['num_imgs', 'offset', 'length']
    if with_categories:
        columns += ['category_id']

    df = pd.DataFrame.from_dict(rows, orient='index')
    df.index.name = 'product_id'
    df.columns = columns
    df.sort_index(inplace=True)
    return df


def make_val_set(df, cat2idx, split_percentage=0.2, drop_percentage=0.):
    # Find the product_ids for each category.
    category_dict = defaultdict(list)
    for ir in tqdm(df.itertuples()):
        category_dict[ir[4]].append(ir[0])

    train_list = []
    val_list = []
    with tqdm(total=len(df)) as pbar:
        for category_id, product_ids in category_dict.items():
            category_idx = cat2idx[category_id]

            # Randomly remove products to make the dataset smaller.
            keep_size = int(len(product_ids) * (1. - drop_percentage))
            if keep_size < len(product_ids):
                product_ids = np.random.choice(product_ids, keep_size, replace=False)

            # Randomly choose the products that become part of the validation set.
            val_size = int(len(product_ids) * split_percentage)
            if val_size > 0:
                val_ids = np.random.choice(product_ids, val_size, replace=False)
            else:
                val_ids = []

            # Create a new row for each image.
            for product_id in product_ids:
                row = [product_id, category_idx]
                for img_idx in range(df.loc[product_id, 'num_imgs']):
                    if product_id in val_ids:
                        val_list.append(row + [img_idx])
                    else:
                        train_list.append(row + [img_idx])
                pbar.update()

    columns = ['product_id', 'category_idx', 'img_idx']
    train_df = pd.DataFrame(train_list, columns=columns)
    val_df = pd.DataFrame(val_list, columns=columns)
    return train_df, val_df


def main():
    parser = argparse.ArgumentParser(description='convert BSON to files')
    parser.add_argument('data_type', choices=['train', 'test'])
    parser.add_argument('--root', '-R', default='data/')
    parser.add_argument('--split_rate', type=float, default=0.2)
    args = parser.parse_args()

    # create category.csv
    categories_in_path = os.path.join(args.root, 'category_names.csv')
    categories_df = pd.read_csv(categories_in_path, index_col='category_id')

    categories_df['category_idx'] = pd.Series(range(len(categories_df)), index=categories_df.index)
    categories_out_path = os.path.join(args.root, 'categories.csv')
    categories_df.to_csv(categories_out_path)
    print('Write to {}.'.format(categories_out_path))

    if args.data_type == 'train':
        # create train_offsets.csv
        train_bson_path = os.path.join(args.root, 'train.bson')
        num_train_products = 7069896
        train_offsets_df = read_bson(train_bson_path, num_train_products, True)

        train_offsets_path = os.path.join(args.root, 'train_offsets.csv')
        train_offsets_df.to_csv(train_offsets_path)
        print('Write to {}.'.format(train_offsets_path))

        # create images.csv
        cat2idx, idx2cat = make_category_tables(categories_df)
        train_images_df, val_images_df = make_val_set(train_offsets_df, cat2idx, args.split_rate)

        train_images_path = os.path.join(args.root, 'train_images.csv')
        train_images_df.to_csv(train_images_path)
        print('Write to {}.'.format(train_images_path))

        val_images_path = os.path.join(args.root, 'val_images.csv')
        val_images_df.to_csv(val_images_path)
        print('Write to {}.'.format(val_images_path))
    else:
        # create test_offsets.csv
        test_bson_path = os.path.join(args.root, 'test.bson')
        num_test_products = 1768182
        test_offsets_df = read_bson(test_bson_path, num_test_products, False)

        test_offsets_path = os.path.join(args.root, 'test_offsets.csv')
        test_offsets_df.to_csv(test_offsets_path)
        print('Write to {}.'.format(test_offsets_path))

        # create images.csv
        # images_df.to_csv(os.path.join(args.root, 'test_images.csv'))


if __name__ == '__main__':
    main()
