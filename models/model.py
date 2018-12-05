import chainer
import chainer.functions as F
import chainer.links as L

class MLP(chainer.Chain):
  def __init__(self, n_in, n_units, n_out):
    super(MLP, self).__init__()
    with self.init_scope():
      self.l1 = L.Linear(n_in, n_units)
      self.l2 = L.Linear(n_units, n_units)
      self.l3 = L.Linear(n_units, n_out)

  def forward(self, x):
    h1 = F.relu(self.l1(x))
    h2 = F.relu(self.l2(h1))
    return self.l3(h2)



class CNN(chainer.Chain):
  def __init__(self):
    super(CNN, self).__init__()
    with self.init_scope():
      self.cn1 = L.Convolution2D(1, 20, 5)
      self.cn2 = L.Convolution2D(20, 50, 5)
      self.fc1 = L.Linear(800, 500)
      self.fc2 = L.Linear(500, 10)

  def forward(self, x):
    h1 = F.max_pooling_2d(F.relu(self.cn1(x)), 2)
    h2 = F.max_pooling_2d(F.relu(self.cn2(h1)), 2)
    h3 = F.dropout(F.relu(self.fc1(h2)))
    return self.fc2(h3)
