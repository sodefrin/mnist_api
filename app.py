from flask import Flask, request, jsonify
app = Flask(__name__)

import chainer
import chainer.links as L
import numpy as np
import cv2

import train_mnist

model = L.Classifier(train_mnist.MLP(784, 1000, 10))
chainer.serializers.load_npz('mnist_model.npz', model)

@app.route('/', methods=['POST'])
def predict():
  file = request.files['image']
  image = cv2.imdecode(np.fromstring(file.stream.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
  image = cv2.resize(image, (28, 28))
  image = image.astype(np.float32).flatten()/255

  x = np.array([image])
  p = model.predictor.forward(x)

  return jsonify({"num": int(p.data.argmax())})

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=8000)
