import argparse
import os

import chainer
from chainer.dataset import convert
# import chainer.functions as F
import chainer.links as L
import pandas as pd


try:
    from dataset import PreprocessedDataset
    from net import LeNet
except Exception:
    raise


def main():
    parser = argparse.ArgumentParser(description='Predict')
    parser.add_argument('dataset', help='Path to test image-label')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--batchsize', '-b', type=int, default=16,
                        help='Number of images in each mini-batch')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--model', '-m', default='model.npz')
    parser.add_argument('--labels', '-l', default='labels.csv')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# minibatch size: {}'.format(args.batchsize))

    # Dataset
    all_files = os.listdir(args.dataset)
    image_files = [(f, f.split('-')[0]) for f in all_files if ('png' in f or 'jpg' in f)]
    dataset = PreprocessedDataset(image_files, args.dataset)
    n_classes = 5270

    labels = pd.read_csv(args.labels)
    label_to_categpry = dict(zip(labels.index, labels['category_id']))

    print('# samples: {}'.format(len(dataset)))
    print('# data shape: {}'.format(dataset[0][0].shape))
    print('# number of label: {}'.format(n_classes))
    print('')

    # Model
    model = L.Classifier(LeNet(n_classes))
    chainer.serializers.load_npz(os.path.join(args.out, args.model), model)
    if args.gpu >= 0:
        model.to_gpu(args.gpu)

    # Iterator
    test_iter = chainer.iterators.SerialIterator(
        dataset, args.batchsize, repeat=False, shuffle=False)

    # Infer
    result = {'_id': [], 'category_id': []}
    for batch in test_iter:
        x_array, _id_array = convert.concat_examples(batch, args.gpu)
        y_array = model.predictor(x_array).data.argmax(axis=1)
        # y_array = F.softmax(model.predictor(x_array)).data.argmax(axis=1)
        l_array = [label_to_categpry[y] for y in y_array]

        result['_id'].extend(list(_id_array))
        result['category_id'].extend(l_array)
    df = pd.DataFrame(
        result['category_id'], result['_id'], columns=['category_id'])
    df.index.name = '_id'
    df = df[~df.index.duplicated()]

    df.to_csv(os.path.join(args.out, 'submission.csv'))


if __name__ == '__main__':
    main()
