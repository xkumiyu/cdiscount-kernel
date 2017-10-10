import argparse
import os

import pandas as pd
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description='make image label list')
    parser.add_argument('--root', '-r', default='data/train/')
    parser.add_argument('--out', '-o', default='data/train_list.txt')
    args = parser.parse_args()

    all_files = os.listdir(args.root)
    categories = [f for f in all_files if os.path.isdir(os.path.join(args.root, f))]
    categories.sort()
    labels = pd.DataFrame(categories)
    labels.index.name = 'label_id'
    labels.rename(columns={0: 'category_id'}, inplace=True)

    labels.to_csv(os.path.join(args.root, '../labels.csv'))

    # label_to_categpry = dict(zip(labels.index, labels['category_id']))
    categpry_to_label = dict(zip(labels['category_id'], labels.index))

    label_list = []
    for category in tqdm(categories):
        category_directory = os.path.join(args.root, category)
        label = categpry_to_label[category]
        for file_name in os.listdir(category_directory):
            file_path = os.path.join(category, file_name)
            label_list.append('{} {}'.format(file_path, label))
    pd.DataFrame(label_list).to_csv(args.out, index=False, header=False)


if __name__ == '__main__':
    main()
