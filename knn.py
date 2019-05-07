import numpy as np
import os
import gzip
from six.moves import urllib
from datetime import datetime


SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'


# 下载mnist数据集，仿照tensorflow的base.py中的写法。
def maybe_download(filename, path, source_url):
    if not os.path.exists(path):
        os.makedirs(path)
    filepath = os.path.join(path, filename)
    print(filepath)
    if not os.path.exists(filepath):
        urllib.request.urlretrieve(source_url, filepath)
    return filepath


maybe_download('train_images', 'data/mnist', SOURCE_URL + TRAIN_IMAGES)
maybe_download('train_labels', 'data/mnist', SOURCE_URL + TRAIN_LABELS)
maybe_download('test_images', 'data/mnist', SOURCE_URL + TEST_IMAGES)
maybe_download('test_labels', 'data/mnist', SOURCE_URL + TEST_LABELS)


# 按32位读取，主要为读校验码、图片数量、尺寸准备的
# 仿照tensorflow的mnist.py写的。
def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


# 抽取图片，并按照需求，可将图片中的灰度值二值化，按照需求，可将二值化后的数据存成矩阵或者张量
# 仿照tensorflow中mnist.py写的
def extract_images(input_file, is_value_binary, is_matrix):
    with gzip.open(input_file, 'rb') as zipf:
        magic = _read32(zipf)
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' % (
                magic, input_file.name))
        num_images = _read32(zipf)
        rows = _read32(zipf)
        cols = _read32(zipf)
        print(input_file)
        print (magic, num_images, rows, cols)
        buf = zipf.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        if is_matrix:
            data = data.reshape(num_images, rows * cols)
        else:
            data = data.reshape(num_images, rows, cols)
        if is_value_binary:
            return np.minimum(data, 1)
        else:
            return data


# 抽取标签
# 仿照tensorflow中mnist.py写的
def extract_labels(input_file):
    with gzip.open(input_file, 'rb') as zipf:
        magic = _read32(zipf)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST label file: %s' % (
                magic, input_file.name))
        num_items = _read32(zipf)
        buf = zipf.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return labels


# 一般的knn分类，跟全部数据同时计算一般距离(欧式距离)，然后找出最小距离的k张图
# 并找出这k张图片的标签，标签占比最大的为newInput的label
def kNNClassify(newInput, dataSet, labels, k):
    numSamples = dataSet.shape[0]               # (60000, 784)
    init_shape = newInput.shape[0]              # (784,)
    newInput = newInput.reshape(1, init_shape)  # (1, 784)

    # np.tile()沿X轴复制1倍（相当于没有复制），再沿Y轴复制numSamples倍
    diff = np.tile(newInput, (numSamples, 1)) - dataSet
    squaredDiff = diff ** 2                    # 平方
    squaredDist = np.sum(squaredDiff, axis=1)  # axis=1每一行的元素相加,将矩阵压缩为一列
    # print(squaredDist)                       # [146 174 172 ... 152 177 165]
    distance = squaredDist ** 0.5              # 开根
    sortedDistIndices = np.argsort(distance)   # 升序的索引
    # print(sortedDistIndices)  # [27059 47003 ... 47146 59439]共60000个

    classCount = {}
    for i in range(k):
        # step 3: choose the min k distance
        voteLabel = labels[sortedDistIndices[i]]

        # step 4: count the times labels occur
        # when the key voteLabel is not in dictionary classCount, get()
        # will return 0
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

    # print(classCount)  # {9:2, 4:1}  {1: 3}.... 每个字典有k个value, key为标签

    # step 5: 返回标签占比最大的为newInput的label
    maxCount = 0
    maxIndex = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex = key
    return maxIndex


# 主函数，先读图片，然后用于测试手写数字
# copy大神http://blog.csdn.net/zouxy09/article/details/16955347的
def testHandWritingClass():
    # step 1: load data
    print("\nstep 1: load data...")
    train_x = extract_images('data/mnist/train_images', True, True)
    train_y = extract_labels('data/mnist/train_labels')
    test_x = extract_images('data/mnist/test_images', True, True)
    test_y = extract_labels('data/mnist/test_labels')

    print(type(train_x))    # <class 'numpy.ndarray'>
    print(type(train_y))
    print(len(train_x[0]))  # 28*28
    print(train_x[:10])     # [[...] [...] [...] ....]
    print(train_y[:10])     # [5 0 4 1 9 2 1 3 1 4]

    # step 2: training...
    print("\nstep 2: training...")
    pass

    # step 3: testing
    print("\nstep 3: testing...")
    start_time = datetime.now()           # 2018-12-03 15:44:23.912004
    numTestSamples = test_x.shape[0]      # (10000, 784)
    matchCount = 0                        # 记录预测正确的个数
    test_num = int(numTestSamples / 10)   # 仅测试test_x数据中的十分之一

    for i in range(test_num):
        predict = kNNClassify(test_x[i], train_x, train_y, 3)
        if predict == test_y[i]:
            matchCount += 1
        if i % 100 == 0:
            print("完成%d张图片" % (i))

    accuracy = float(matchCount) / test_num
    end_time = datetime.now()
    print("一共运行了%d秒" % ((end_time - start_time).seconds))

    # step 4: show the result
    print("\nstep 4: show the result...")
    print('\nThe classify accuracy is: %.2f%%' % (accuracy * 100))


if __name__ == '__main__':
    testHandWritingClass()
    print('OK')
