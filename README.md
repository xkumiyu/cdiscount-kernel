Chainer kernel for [Cdiscount’s Image Classification Challenge](https://www.kaggle.com/c/cdiscount-image-classification-challenge) Kaggle competition.

# Requirements

* Chainer 3.0.0

# Preparation

## Downloading data

Download data using [kaggle-cli](https://github.com/floydwch/kaggle-cli).

``` sh
$ kg download -u <username> -p <password> -c cdiscount-image-classification-challenge
```

Extract the file.

``` sh
$ 7z x category_names.7z
```

## Data conversion

Convert BSON to jpeg file.

``` sh
$ util/convert_BSON_to_files.py -d train -r <data directory>
$ util/convert_BSON_to_files.py -d test -r <data directory>
```

category_names.csv, train.bson, test.bson is necessary in <data directory>.

* File pattern to be converted
  * train files: `<data directory>/train/<category>/<_id>-<index>.jpg`
  * test files: `<data directory>/test/<_id>-<index>.jpg`

This script referred to [this notebook](https://www.kaggle.com/bguberfain/not-so-naive-way-to-convert-bson-to-files).

## Make image label list

``` sh
$ python util/make_image_label_list.py
```

# Training

``` sh
$ python train.py <train data list>
```

# Inference

``` sh
$ python infer.py <test data directory>
```

# Appendix

* 5,270 different categories
* image size: 180 x 180

## Train Data

* 7,069,896 products
* train.bson: 59GB

* 12,371,293 files
* image files: 81GB

## Test Data

* 1,768,182 products
* test.bson: 15GB

* 3,095,080 files
* image files: 21GB

---

files: 0.86055 iters/sec.
