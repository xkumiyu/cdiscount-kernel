import chainer
import chainer.functions as F
import chainer.links as L


class LeNet(chainer.Chain):

    def __init__(self, class_labels=10):
        super(LeNet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 6, ksize=5)
            self.conv2 = L.Convolution2D(None, 16, ksize=5)
            self.fc3 = L.Linear(None, 120)
            self.fc4 = L.Linear(None, 84)
            self.fc5 = L.Linear(None, class_labels)

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.conv2(x))
        h = F.max_pooling_2d(h, 2, stride=2)
        h = self.fc3(h)
        h = self.fc4(h)
        h = self.fc5(h)

        return h
