import argparse
import os
import sys
import threading

import matplotlib
try:
    matplotlib.use('Agg')
except Exception:
    raise

import chainer
import chainer.links as L
from chainer.training import extensions
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/model')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/util')

try:
    from create_lookup_tables import make_category_tables
    from dataset import DatasetwithBSON
    from dataset import DatasetwithJPEG
    from lenet import LeNet
    from resnet import ResNet152
    from vgg import VGG
except Exception:
    raise


def setup_dataset_with_bson(root):
    train_bson_path = os.path.join(root, 'train_example.bson')
    categories_df = pd.read_csv(os.path.join(root, 'categories.csv'), index_col=0)
    cat2idx, idx2cat = make_category_tables(categories_df)

    offsets_df = pd.read_csv(os.path.join(root, 'train_offsets.csv'), index_col=0)
    train_images_df = pd.read_csv(os.path.join(root, 'train_images.csv'), index_col=0)
    val_images_df = pd.read_csv(os.path.join(root, 'val_images.csv'), index_col=0)

    bson_file = open(train_bson_path, 'rb')
    lock = threading.Lock()

    train = DatasetwithBSON(bson_file, train_images_df, offsets_df, cat2idx, lock)
    val = DatasetwithBSON(bson_file, val_images_df, offsets_df, cat2idx, lock)

    return train, val


def setup_dataset_with_jpeg(root, listfile, split_rate, seed):
    dataset = DatasetwithJPEG(listfile, root)

    split_at = int(len(dataset) * split_rate)
    train, val = chainer.datasets.split_dataset_random(dataset, split_at, seed)

    return train, val


def main():
    archs = {
        'lenet': LeNet,
        'vgg': VGG,
        'resnet152': ResNet152,
    }

    parser = argparse.ArgumentParser(description='Kaggle Kernel')
    parser.add_argument('--listfile', '-l', help='Path to training image-label list file')
    parser.add_argument('--arch', '-a', default='lenet',
                        choices=['lenet', 'vgg', 'resnet152'])
    parser.add_argument('--batchsize', '-b', type=int, default=16,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--root', '-R', default='.',
                        help='Root directory path of image files')
    parser.add_argument('--split_rate', type=float, default=0.99)
    parser.add_argument('--snapshot_interval', type=int, default=10000,
                        help='Interval of snapshot')
    parser.add_argument('--display_interval', type=int, default=100,
                        help='Interval of displaying log to console')
    parser.add_argument('--log_interval', type=int, default=1000,
                        help='Interval of log to output')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# minibatch size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))

    # Dataset
    # train, val = setup_dataset_with_bson(args.root)
    train, val = setup_dataset_with_jpeg(args.root, args.listfile, args.split_rate, args.seed)
    n_classes = 5270

    print('# iterators / epoch: {}'.format(int(len(train) / args.batchsize)))
    print('# train samples: {}'.format(len(train)))
    print('# validation samples: {}'.format(len(val)))
    print('# data shape: {}'.format(train[0][0].shape))
    print('# number of label: {}'.format(n_classes))

    # Iterator
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    val_iter = chainer.iterators.SerialIterator(
        val, args.batchsize, repeat=False, shuffle=False)

    # Model
    model = L.Classifier(archs[args.arch](n_classes))
    print('# model: {}'.format(args.arch))

    if args.gpu >= 0:
        model.to_gpu(args.gpu)

    # Updater
    # optimizer = chainer.optimizers.Adam(alpha=1e-4)
    optimizer = chainer.optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    optimizer.setup(model)

    updater = chainer.training.StandardUpdater(
        train_iter, optimizer, device=args.gpu)

    # Trainer
    trainer = chainer.training.Trainer(
        updater, (args.epoch, 'epoch'), out=args.out)

    snapshot_interval = (args.snapshot_interval, 'iteration')
    display_interval = (args.display_interval, 'iteration')
    log_interval = (args.log_interval, 'iteration')

    trainer.extend(extensions.Evaluator(val_iter, model, device=args.gpu), trigger=log_interval)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.PrintReport([
        'epoch', 'main/loss', 'main/accuracy',
        'validation/main/loss', 'validation/main/accuracy', 'elapsed_time']),
        trigger=display_interval)
    trainer.extend(extensions.PlotReport([
        'main/accuracy', 'validation/main/accuracy'],
        trigger=log_interval, file_name='accuracy.png'))
    trainer.extend(
        extensions.snapshot(filename='snapshot_iter_{.updater.iteration}.npz'),
        trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        model, 'model_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run
    trainer.run()

    # Save model
    if args.gpu >= 0:
        model.to_cpu()
    chainer.serializers.save_npz(os.path.join(args.out, 'model.npz'), model)

    print('Finished!')


if __name__ == '__main__':
    main()
