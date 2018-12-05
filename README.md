## MNIST API

mnistのchainerでの学習＆モデルの保存＆画像を判定するAPIのサンプル

### インストール

```
$ pipenv install
```

### モデルの学習

#### MLP

```
$ pipenv run mlp
```

#### CNN

```
$ pipenv run cnn
```

### API立ち上げ

```
$ pipenv run api
```

### curlでのテスト

```
$ curl -X POST localhost:8000 -F "image=@0.png"
```
