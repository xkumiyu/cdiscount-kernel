import io

import bson
import chainer
from chainer.links.model.vision.resnet import prepare
import numpy as np
from PIL import Image


class DatasetwithBSON(chainer.dataset.DatasetMixin):

    def __init__(self, bson_file, images_df, offsets_df, cat2idx, lock, dtype=np.float32):
        self.bson_file = bson_file
        self.images_df = images_df
        self.offsets_df = offsets_df
        self.cat2idx = cat2idx
        self.lock = lock

        self._dtype = dtype
        self._label_dtype = np.int32

    def __len__(self):
        return len(self.images_df)

    def get_example(self, i):
        with self.lock:
            image_row = self.images_df.iloc[i]
            product_id = image_row['product_id']
            offset_row = self.offsets_df.loc[product_id]

            self.bson_file.seek(offset_row['offset'])
            item_data = self.bson_file.read(offset_row['length'])
        item = bson.BSON.decode(item_data)
        img_idx = image_row['img_idx']
        bson_img = item['imgs'][img_idx]['picture']

        img = Image.open(io.BytesIO(bson_img))
        img = np.asarray(img, dtype=self._dtype).transpose(2, 0, 1)
        img = img / 255.

        label = self.cat2idx[item['category_id']]
        label = np.array(label, dtype=self._label_dtype)

        return img, label


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
