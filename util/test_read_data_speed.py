import argparse
import os
import threading
import time

import chainer
import pandas as pd

try:
    from create_lookup_tables import make_category_tables
    from dataset import PreprocessedDataset
except Exception:
    raise


class PreprocessedImageDataset(chainer.dataset.DatasetMixin):

    def __init__(self, path, root):
        self.base = chainer.datasets.LabeledImageDataset(path, root)

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        image, label = self.base[i]
        image = image / 255.
        return image, label


def prepare_bson(args):
    # Dataset
    train_bson_path = os.path.join(args.root, 'train_example.bson')
    categories_df = pd.read_csv(os.path.join(args.root, 'categories.csv'), index_col=0)
    cat2idx, idx2cat = make_category_tables(categories_df)

    offsets_df = pd.read_csv(os.path.join(args.root, 'train_offsets.csv'), index_col=0)
    train_images_df = pd.read_csv(os.path.join(args.root, 'train_images.csv'), index_col=0)
    val_images_df = pd.read_csv(os.path.join(args.root, 'val_images.csv'), index_col=0)

    bson_file = open(train_bson_path, 'rb')
    lock = threading.Lock()

    train = PreprocessedDataset(bson_file, train_images_df, offsets_df, cat2idx, lock)
    val = PreprocessedDataset(bson_file, val_images_df, offsets_df, cat2idx, lock)

    # Iterator
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(
        val, args.batchsize, repeat=False, shuffle=False)

    return train_iter, test_iter


def prepare_jepg(args):
    # Dataset
    root = os.path.join(args.root, 'train_example')
    dataset = PreprocessedImageDataset(args.dataset, root)

    split_at = int(len(dataset) * 0.9)
    train, val = chainer.datasets.split_dataset_random(dataset, split_at, 0)

    # Iterator
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(
        val, args.batchsize, repeat=False, shuffle=False)

    return train_iter, test_iter


def loop(args, train_iter):
    n = 0
    while train_iter.epoch < args.epoch:
        batch = train_iter.next()
        n += len(batch)
    return n


def main():
    parser = argparse.ArgumentParser(description='Kaggle Kernel')
    parser.add_argument('--dataset', '-d', default='data/train_list.txt')
    parser.add_argument('--root', '-R', default='data/',
                        help='Root directory path of image files')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--batchsize', '-b', type=int, default=16,
                        help='Number of images in each mini-batch')
    args = parser.parse_args()

    t1a = time.time()
    train_iter, _ = prepare_bson(args)
    t1b = time.time()
    print('[prepare bson] time: {:.2f} sec'.format(t1b - t1a))

    n = loop(args, train_iter)
    t1 = time.time() - t1b
    print('[loop bson] time: {:.2f} sec, {:.4f} sec/images'.format(t1, t1 / n))

    t2a = time.time()
    train_iter, _ = prepare_jepg(args)
    t2b = time.time()
    print('[prepare jpeg] time: {:.2f} sec'.format(t2b - t2a))

    n = loop(args, train_iter)
    t2 = time.time() - t2b
    print('[loop jpeg] time: {:.2f} sec, {:.4f} sec/images'.format(t2, t2 / n))


if __name__ == '__main__':
    main()
