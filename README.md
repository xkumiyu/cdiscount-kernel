Chainer kernel for [Cdiscount’s Image Classification Challenge](https://www.kaggle.com/c/cdiscount-image-classification-challenge) Kaggle competition.

# Preparation

## Downloading data

Download data using [kaggle-cli](https://github.com/floydwch/kaggle-cli).

``` sh
$ kg download -u <username> -p <password> -c cdiscount-image-classification-challenge
```

## Data conversion

Convert BSON to jpeg file. It takes several hours to convert.

``` sh
$ convert_BSON_to_files.py -d train -r <data directory>
$ convert_BSON_to_files.py -d test -r <data directory>
```

category_names.csv, train.bson, test.bson is necessary in <data directory>.

* File pattern to be converted
  * train files: `<data directory>/train/<category>/<_id>-<index>.jpg`
  * test files: `<data directory>/test/<_id>-<index>.jpg`

This script referred to [this notebook](https://www.kaggle.com/bguberfain/not-so-naive-way-to-convert-bson-to-files).

## Make image label list

``` sh
$ python make_image_label_list.py
```

# Training

``` sh
$ python train.py
```

# Inference

``` sh
$ python infer.py
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
