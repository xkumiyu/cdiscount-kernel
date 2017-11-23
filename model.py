import chainer
import chainer.links as L


class ResNet152(chainer.Chain):
    def __init__(self, class_labels):
        super(ResNet152, self).__init__()

        with self.init_scope():
            self.base = L.ResNet152Layers()
            self.fc6 = L.Linear(2048, class_labels)

    def __call__(self, x):
        h = self.base(x, layers=['pool5'])['pool5']
        return self.fc6(h)
