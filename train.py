import argparse
import os

import chainer
import chainer.links as L
from chainer.training import extensions

try:
    from dataset import DatasetfromMongoDB
    from net import LeNet
except Exception:
    raise


def main():
    parser = argparse.ArgumentParser(description='Chainer example: DCGAN')
    parser.add_argument('--batchsize', '-b', type=int, default=50,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=1000,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--db_name', '-d', default='cicc')
    parser.add_argument('--col_name', '-c', default='train_example')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--split_rate', type=float, default=0.8)
    parser.add_argument('--snapshot_interval', type=int, default=1000,
                        help='Interval of snapshot')
    parser.add_argument('--display_interval', type=int, default=100,
                        help='Interval of displaying log to console')
    args = parser.parse_args()

    # Dataset
    dataset = DatasetfromMongoDB(db_name=args.db_name, col_name=args.col_name)
    n_classes = len(dataset.get_labels())

    split_at = int(len(dataset) * args.split_rate)
    train, test = chainer.datasets.split_dataset(dataset, split_at)

    print('GPU: {}'.format(args.gpu))
    print('# train samples: {}'.format(len(train)))
    print('# test samples: {}'.format(len(test)))
    print('# data shape: {}'.format(train[0][0].shape))
    print('# number of label: {}'.format(n_classes))

    # Iterator
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(
        test, args.batchsize, repeat=False, shuffle=False)

    # Model
    model = L.Classifier(LeNet(n_classes))

    if args.gpu >= 0:
        model.to_gpu(args.gpu)

    # Updater
    optimizer = chainer.optimizers.Adam(alpha=1e-4)
    optimizer.setup(model)

    updater = chainer.training.StandardUpdater(
        train_iter, optimizer, device=args.gpu)

    # Trainer
    trainer = chainer.training.Trainer(
        updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport([
        'epoch', 'main/loss', 'main/accuracy',
        'validation/main/loss', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
    trainer.extend(extensions.ProgressBar(update_interval=10))

    # Run
    trainer.run()

    # Save generator model
    if args.gpu >= 0:
        model.to_cpu()
    chainer.serializers.save_npz(os.path.join(args.out, 'model.npz'), model)


if __name__ == '__main__':
    main()