import chainer
from chainer.links.model.vision.resnet import prepare
import random


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

    def random_crop(self, image):
        _, h, w = image.shape
        crop_size = random.randint(0, h // 2)
        top = random.randint(0, h - crop_size - 1)
        left = random.randint(0, w - crop_size - 1)
        if random.randint(0, 1):
            image = image[:, :, ::-1]
        bottom = top + crop_size
        right = left + crop_size
        image = image[:, top:bottom, left:right]
        return image
