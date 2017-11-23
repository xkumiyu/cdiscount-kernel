import chainer
from chainer.links.model.vision.resnet import prepare


class DatasetwithJPEG(chainer.dataset.DatasetMixin):

    def __init__(self, path, root):
        self.base = chainer.datasets.LabeledImageDataset(path, root)

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        image, label = self.base[i]
        image = prepare(image)
        # image = image / 255.
        return image, label
