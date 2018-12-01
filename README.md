## MNIST API

mnistのchainerでの学習＆モデルの保存＆画像を判定するAPIのサンプル

### インストール

```
$ pip install -r requirements.txt
```

### モデルの学習

```
$ python train_mnist.py
```

### API立ち上げ

```
$ python app.py
```

### curlでのテスト

```
$ curl -X POST localhost:8000 -F "image=@0.png"
```
