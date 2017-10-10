import argparse
import os

import chainer
from chainer.dataset import convert
# import chainer.functions as F
import chainer.links as L
from tqdm import tqdm


try:
    from dataset import DatasetwithJPEG
    from net import LeNet
except Exception:
    raise


def setup_dataset_with_jpeg(root, listfile, split_rate, seed):
    dataset = DatasetwithJPEG(listfile, root)

    split_at = int(len(dataset) * split_rate)
    train, val = chainer.datasets.split_dataset_random(dataset, split_at, seed)

    return train, val


def main():
    parser = argparse.ArgumentParser(description='Predict')
    parser.add_argument('--listfile', '-l', help='Path to training image-label list file')
    parser.add_argument('--batchsize', '-b', type=int, default=16,
                        help='Number of images in each mini-batch')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--root', '-R', default='.',
                        help='Root directory path of image files')
    parser.add_argument('--split_rate', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model', '-m', default='model.npz')
    parser.add_argument('--drop_rate', type=float, default=0.)
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# minibatch size: {}'.format(args.batchsize))

    # Dataset
    _, val = setup_dataset_with_jpeg(args.root, args.listfile, args.split_rate, args.seed)
    if args.drop_rate > 0:
        split_at = int(len(val) * args.drop_rate)
        _, val = chainer.datasets.split_dataset_random(val, split_at)
    n_classes = 5270

    print('# validation samples: {}'.format(len(val)))
    print('# data shape: {}'.format(val[0][0].shape))
    print('# number of label: {}'.format(n_classes))
    print('')

    # Model
    model = L.Classifier(LeNet(n_classes))
    chainer.serializers.load_npz(os.path.join(args.out, args.model), model)
    if args.gpu >= 0:
        model.to_gpu(args.gpu)

    # Iterator
    val_iter = chainer.iterators.SerialIterator(
        val, args.batchsize, repeat=False, shuffle=False)
    n_iter = int(len(val) / args.batchsize)

    # Validate
    sum_loss = 0
    sum_accuracy = 0
    pbar = tqdm(total=n_iter)
    for batch in val_iter:
        x_array, t_array = convert.concat_examples(batch, args.gpu)
        x = chainer.Variable(x_array)
        t = chainer.Variable(t_array)
        loss = model(x, t)
        sum_loss += float(loss.data) * len(t.data)
        sum_accuracy += float(model.accuracy.data) * len(t.data)
        pbar.update()
    pbar.close()

    print('val mean loss: {:4f}, accuracy: {:4f}'.format(
        sum_loss / len(val), sum_accuracy / len(val)))


if __name__ == '__main__':
    main()
