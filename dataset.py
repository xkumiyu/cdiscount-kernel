import chainer
from chainer.links.model.vision.resnet import prepare
import numpy as np


class DatasetwithJPEG(chainer.dataset.DatasetMixin):

    def __init__(self, path, root, crop=False):
        self.base = chainer.datasets.LabeledImageDataset(path, root)
        self.crop = crop

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        image, label = self.base[i]
        if self.crop:
            image = self.random_crop(image)
        image = prepare(image)
        return image, label

    def random_crop(self, image, rate=0.5):
        if np.random.rand() < rate:
            image = image[:, :, ::-1]
        if np.random.rand() < rate:
            return image
        _, h, w = image.shape
        crop_size = np.random.randint(h // 2, h)
        top = np.random.randint(0, h - crop_size)
        left = np.random.randint(0, w - crop_size)

        bottom = top + crop_size
        right = left + crop_size
        image = image[:, top:bottom, left:right]
        return image
