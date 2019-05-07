import numpy as np
import os
import gzip
from datetime import datetime
import random
import struct


def readfile():
    with open('data/train-images.idx3-ubyte', 'rb') as f1:
        buf1_tran = f1.read()
    with open('data/train-labels.idx1-ubyte', 'rb') as f2:
        buf2_tran = f2.read()
    with open('data/t10k-images.idx3-ubyte', 'rb') as f3:
        buf1_test = f3.read()
    with open('data/t10k-labels.idx1-ubyte', 'rb') as f4:
        buf2_test = f4.read()
    return buf1_tran, buf2_tran, buf1_test, buf2_test


def get_image(buf):
    head = struct.unpack_from('>IIII', buf, 0)  # 取前4个整数，返回一个元组
    offset = struct.calcsize('>IIII')  # 定位到data开始的位置
    img_num = head[1]
    width = head[2]
    height = head[3]
    bits = img_num * width * height  # data一共有60000*28*28个像素值
    bits_string = '>' + str(bits) + 'B'  # fmt格式：'>47040000B'
    imgs = struct.unpack_from(bits_string, buf, offset)  # 取data数据，返回一个元组
    imgs = np.reshape(imgs, [img_num, width * height]
                      )  # reshape为[60000,784]型数组
    return imgs, head


def get_label(buf):  # 得到标签数据
    head = struct.unpack_from('>II', buf, 0)  # 取label文件前2个整形数
    label_num = head[1]
    offset = struct.calcsize('>II')  # 定位到label数据开始的位置
    num_string = '>' + str(label_num) + "B"    # fmt格式：'>60000B'
    labels = struct.unpack_from(num_string, buf, offset)  # 取label数据
    labels = np.reshape(labels, [label_num])   # 转型为列表(一维数组)
    return labels, head


def test_sk_kmeans(data_set, label_set):
    # 调用sklearn的kmean
    from sklearn.cluster import KMeans
    k = 10
    # 'k-means++' or 'random' or an ndarray
    kmeans = KMeans(n_clusters=k, init='random')
    kmeans.fit(data_set)
    cluster_index = {}
    for i in range(k):
        cluster_index[i] = []
    for i in range(len(kmeans.labels_)):
        for lab in range(k):
            if kmeans.labels_[i] == lab:
                cluster_index[lab].append(i)

    test_accuracy(cluster_index, label_set)


def test_accuracy(cluster_index, label_set):
    # 统计准确率
    k = 10
    right_num = 0
    error_num = 0
    for i in range(k):
        print('第', i + 1, '聚类个数：', len(cluster_index[i]))
        every_cluster = {}  # 保存这个聚类中数字1-9的数量
        for x in range(k):  # 初始化
            every_cluster[x] = 0
        for j in cluster_index[i]:  # 统计这个聚类中数字1-9的数量
            for y in range(k):
                if label_set[j] == y:
                    every_cluster[y] = every_cluster[y] + 1
        print(every_cluster)
        lab = max(every_cluster, key=every_cluster.get)
        print('这个聚类可能是数字', lab, '个数为', every_cluster[lab],
              '错误个数', len(cluster_index[i]) - every_cluster[lab], '\n')
        right_num = right_num + every_cluster[lab]
        error_num = error_num + len(cluster_index[i]) - every_cluster[lab]
    print('正确个数：', right_num)
    print('错误个数', error_num)
    print('正确率 : {0:.2f}%'.format(
        (right_num * 1.0 / (error_num + right_num) * 1.0) * 100))
    print('错误率 : {0:.2f}%'.format(
        (error_num * 1.0 / (error_num + right_num) * 1.0) * 100))


# 打印聚类效果
def test_accuracy(cluster_index, label_set):
    # 统计准确率
    k = 10
    right_num = 0
    error_num = 0
    for i in range(k):
        print('第', i + 1, '聚类个数：', len(cluster_index[i]))
        every_cluster = {}  # 保存这个聚类中数字1-9的数量
        for x in range(k):  # 初始化
            every_cluster[x] = 0
        for j in cluster_index[i]:  # 统计这个聚类中数字1-9的数量
            for y in range(k):
                if label_set[j] == y:
                    every_cluster[y] = every_cluster[y] + 1
        print(every_cluster)
        lab = max(every_cluster, key=every_cluster.get)
        print('这个聚类可能是数字', lab, '个数为', every_cluster[lab],
              '错误个数', len(cluster_index[i]) - every_cluster[lab], '\n')
        right_num = right_num + every_cluster[lab]
        error_num = error_num + len(cluster_index[i]) - every_cluster[lab]
    print('正确个数：', right_num)
    print('错误个数', error_num)
    print('正确率 : {0:.2f}%'.format(
        (right_num * 1.0 / (error_num + right_num) * 1.0) * 100))
    print('错误率 : {0:.2f}%'.format(
        (error_num * 1.0 / (error_num + right_num) * 1.0) * 100))


# 主函数，读图片测试手写数字
def hand_writing():
    # step 1: load data
    print("\nstep 1: load data...")
    train_images, train_labels, t10k_images, t10k_labels = readfile()
    train_imgs, train_data_head = get_image(train_images)
    train_label, train_label_head = get_label(train_labels)
    t10k_imgs, test_data_head = get_image(t10k_images)
    t10k_label, test_label_head = get_label(t10k_labels)
    print(train_data_head)   # (2051, 60000, 28, 28)
    print(train_label_head)  # (2049, 60000)
    print(test_data_head)    # (2051, 10000, 28, 28)
    print(test_label_head)   # (2049, 10000)

    # 合并数据集
    all_imgs = np.concatenate((train_imgs, t10k_imgs), axis=0)
    print(all_imgs.shape)  # (70000, 784)
    print(type(all_imgs))  # <class 'numpy.ndarray'>
    # print_data(train_imgs, train_label, 10)
    all_labels = np.concatenate((train_label, t10k_label))
    print(all_labels.shape)  # (70000,)

    # 随机抽取50000个
    n = 50000
    index = random.sample(range(0, all_imgs.shape[0]), n)
    data_set = []
    label_set = []
    for i in index:
        data_set.append(all_imgs[i])
        label_set.append(all_labels[i])
    print(type(data_set))   # <class 'list'>
    print(len(data_set))    # 50000
    print(type(label_set))  # <class 'list'>
    print(len(label_set))   # 50000
    # print_data(data_set, label_set, 10)

    # step 2: training...
    print("\nstep 2: training...")
    # 测试sklearn库中的kmeans
    # test_sk_kmeans(data_set, label_set)

    # 调用skfuzzy的cmean
    from skfuzzy.cluster import cmeans
    # c聚类个数，m是隶属度加权指数，maxiter迭代次数
    # center聚类中心，u模糊c分区矩阵, u0模糊c分区矩阵的初始猜测,
    # d欧几里德距离矩阵, jm目标函数，p运行的迭代次数，fpc模糊分配系数
    # 这里的data注意shape（特征数目，数据个数），与很多训练数据的shape正好是相反的。
    center, u, u0, d, jm, p, fpc = cmeans(
        np.array(data_set).T, m=2, c=10, error=0.005, maxiter=1000)
    print(u.shape)     # (10, 2000)
    print(type(u))     # <class 'numpy.ndarray'>
    # print(len(jm))     # [....]
    # print(p)           # 每次迭代有一次目标函数
    # print(fpc)         # 系数

    # step 3: 打印效果，将样本最大概率聚类
    cluster_index = {}  # 保存 k 个聚类索引号
    for i in range(10):
        cluster_index[i] = []  # 初始化聚类索引
    for j in range(u.shape[1]):
        max_u = 0
        lab = 0
        for i in range(u.shape[0]):
            if u[i][j] > max_u:       # 找出每列中概率最大的
                lab = i
                max_u = u[i][j]
        cluster_index[lab].append(j)         # 将该样本索引聚类
    test_accuracy(cluster_index, label_set)  # 打印聚类效果


if __name__ == '__main__':
    hand_writing()
    print('OK')
