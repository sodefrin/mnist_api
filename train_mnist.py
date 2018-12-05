import argparse

import chainer
import chainer.links as L
from chainer import training
from chainer.training import extensions

import models.model as m

def main():
  parser = argparse.ArgumentParser(description='Chainer example: MNIST')
  parser.add_argument('--alg', dest='alg', help='mlp cnn')
  args = parser.parse_args()

  print('alg: {}'.format(args.alg))
  print('')

  if args.alg == 'mlp':
    model = L.Classifier(m.MLP(784, 1000, 10))
    train, test = chainer.datasets.get_mnist()
  elif args.alg == 'cnn':
    model = L.Classifier(m.CNN())
    train, test = chainer.datasets.get_mnist(ndim=3)

  chainer.backends.cuda.get_device_from_id(0).use()

  optimizer = chainer.optimizers.Adam()
  optimizer.setup(model)


  train_iter = chainer.iterators.SerialIterator(train, 100)
  test_iter = chainer.iterators.SerialIterator(test, 100, repeat=False, shuffle=False)

  updater = training.updaters.StandardUpdater( train_iter, optimizer, device=0)
  trainer = training.Trainer(updater, (20, 'epoch'))
  trainer.extend(extensions.Evaluator(test_iter, model, device=0))
  trainer.extend(extensions.dump_graph('main/loss'))

  frequency = 20
  trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))
  trainer.extend(extensions.LogReport())

  trainer.extend(extensions.PrintReport(
      ['epoch', 'main/loss', 'validation/main/loss',
       'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

  trainer.extend(extensions.ProgressBar())

  trainer.run()

  model.to_cpu()
  chainer.serializers.save_npz('mnist_model.npz', model)


if __name__ == '__main__':
  main()
