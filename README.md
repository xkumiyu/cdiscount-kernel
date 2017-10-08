Chainer kernel for Cdiscount’s Image Classification Challenge Kaggle competition.

# Requirements

* python
* chainer
* kaggle-cli
* mongo

# Preparation

## Downlaod Data

``` sh
$ kg download -u <username> -p <password> -c cdiscount-image-classification-challenge
```

## Setup MongoDB

``` sh
$ sudo mongod --dbpath mongodb
$ mongorestore -v --db cicc train.bson
$ mongorestore -v --db cicc test.bson
```

# Training

``` sh
$ python train.py
```

# Inference

``` sh
$ python infer.py
```
