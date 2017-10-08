import io

import chainer
from chainer.dataset import dataset_mixin
import numpy as np
from PIL import Image
from pymongo import MongoClient


class DatasetfromMongoDB(dataset_mixin.DatasetMixin):
    def __init__(self, db_name='cicc', col_name='train', dtype=np.float32):
        self._dtype = dtype
        self._label_dtype = np.int32

        client = MongoClient('localhost', 27017)
        db = client[db_name]
        self.col = db[col_name]
        self.examples = list(self.col.find({}, {'imgs': 0}))
        self.labels = self.get_labels()

    def __len__(self):
        return len(self.examples)

    def get_labels(self):
        category_ids = [e['category_id'] for e in self.examples]
        return {cid: i for i, cid in enumerate(list(set(category_ids)))}

    def get_example(self, i):
        _id = self.examples[i]['_id']
        doc = self.col.find_one({'_id': _id})

        img = doc['imgs'][0]['picture']
        img = Image.open(io.BytesIO(img))
        img = np.asarray(img, dtype=self._dtype).transpose(2, 0, 1)
        img = img / 255.

        if chainer.config.train:
            label = self.labels[doc['category_id']]
            label = np.array(label, dtype=self._label_dtype)

            return img, label
        else:
            return img, _id
