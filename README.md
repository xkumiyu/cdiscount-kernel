kernel for Cdiscount’s Image Classification Challenge Kaggle competition.

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
$ sudo mongod --dbpath data/mongodb --logpath data/mongodb.log
$ mongorestore -v --db cicc data/train.bson
$ mongorestore -v --db cicc data/test.bson
```

# train

``` sh
$ python train.py
```
