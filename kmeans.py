import numpy as np
import struct
import random


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


# kmeans
def kmeans_classify(data_set, data_label, k):
    center = {}            # 保存 k 个中心
    for i in range(k):
        center[i] = np.random.randint(0, 256, data_set.shape[1])  # 随机 k 个中心点

    n = 50  # 迭代次数
    for i in range(n):
        # 更新距离
        distance = update_distance(data_set, center, k)
        # 重新聚类
        cluster_index = update_cluster(distance, data_label, k)
        # 更新中心
        center = update_center(data_set, cluster_index, k)


# kmeans++
def kmeans_plus_classify(data_set, data_label, k):
    center = {}            # 保存 k 个中心
    num_sample = data_set.shape[0]

    # 随机一个样本作为第一个中心
    i = np.random.randint(0, num_sample, 1)
    center[0] = data_set[i[0]]
    print('0 号中心为样本', i[0], '标签为', data_label[i[0]])

    for i in range(k - 1):             # 每次循环找最远距离的样本做中心
        min_distance_list = []       # 保存每个样本与当前已有中心之间的最短距离

        distance = update_distance(data_set, center, i + 1)
        for j in range(num_sample):  # 筛选出每个样本与当前已有中心之间的最短距离
            min_d = 99999
            for r in distance:
                if min_d > distance[r][j]:
                    min_d = distance[r][j]
            min_distance_list.append(min_d)

        sorted_dist_indices = np.argsort(-np.array(min_distance_list))  # 降序的索引
        # print(sortedDistIndices)  # [27059 47003 ... 47146 59439]共60000个

        # 最远距离的样本做另一个中心
        center[i + 1] = data_set[sorted_dist_indices[0]]
        print(i + 1, '号中心为样本',
              sorted_dist_indices[0], '标签为', data_label[sorted_dist_indices[0]])

    # 中心选举完成，开始更新中心
    n = 50     # 迭代次数
    for i in range(n):
        # 更新距离
        distance = update_distance(data_set, center, k)
        # 重新聚类
        cluster_index = update_cluster(distance, data_label, k)
        # 更新中心
        center = update_center(data_set, cluster_index, k)


# 计算距离
def update_distance(data_set, center, k):
    distance = {}  # 保存 k 个中心到每个图片的距离
    for i in range(k):
        # 计算每个图片与中心点的距离
        num_samples = data_set.shape[0]
        # np.tile()沿X轴复制1倍（相当于没有复制），再沿Y轴复制numSamples倍
        diff = np.tile(center[i], (num_samples, 1)) - data_set
        squared_diff = diff ** 2  # 平方
        squared_diff = np.sum(squared_diff, axis=1)   # axis=1每一行的元素相加,将矩阵压缩为一列
        # print(squaredDist)                          # [146 174 172 ... 152 177 165] 共60000个
        distance[i] = squared_diff ** 0.5  # 开根
    # print(distance)     # { 0: array([19.79898987,20.24845673, ...])  1:.......  }
    return distance


# 聚类
def update_cluster(distance, data_label, k):
    cluster_index = {}  # 保存 k 个聚类索引号
    for i in range(k):
        cluster_index[i] = []  # 初始化聚类索引
    for j in range(len(distance[0])):
        min_dist = 99999
        lab = 0
        for i in range(k):
            if distance[i][j] < min_dist:  # i行的距离最小，则聚类到i
                min_dist = distance[i][j]
                lab = i
            else:
                continue
        cluster_index[lab].append(j)       # 保存距离最小的

    # 打印每个聚类中数字0-9的个数
    for i in range(k):
        print('聚类 ', i, ' 的个数：', len(cluster_index[i]))
        # 计算每个聚类中，数字1，2，3,....的个数
        every_cluster = {}    # 保存数量
        for x in range(k):
            every_cluster[x] = 0
        for j in cluster_index[i]:
            for y in range(k):
                if data_label[j] == y:
                    every_cluster[y] = every_cluster[y] + 1
        print(every_cluster)

    # 打印效果
    test_accuracy(cluster_index, data_label)
    return cluster_index


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


# 更新中心
def update_center(data_set, cluster_index, k):
    center = {}
    for i in range(k):
        center[i] = []
        all_sum = 0
        print('更新聚类 ', i)
        if len(cluster_index[i]) == 0:    # 如果某个聚类，没有元素，重新随机一个
            # center[i] = np.random.randint(0, 256, data_set.shape[1])
            print('出错！')
        else:
            for j in cluster_index[i]:
                all_sum = data_set[j] + all_sum
            center[i] = all_sum / len(cluster_index[i])
    return center


# 打印前n个数据看看
def print_data(data_set, data_label, n):
    for i in range(n):
        for j in range(28):
            for n in data_set[i][28 * j:28 * (j + 1)]:
                print('%03d' % n, end=' ')
            print('')
        print('标签：', data_label[i])


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
    print(all_imgs.shape)   # (70000, 784)
    print(type(all_imgs))   # <class 'numpy.ndarray'>
    # print_data(train_imgs, train_label, 10)
    all_labels = np.concatenate((train_label, t10k_label))
    print(all_labels.shape)   # (70000,)

    # 随机抽取50000个
    n = 50000
    index = random.sample(range(0, all_imgs.shape[0]), n)
    data_set = []
    label_set = []
    for i in index:
        data_set.append(all_imgs[i])
        label_set.append(all_labels[i])
    print(type(data_set))    # <class 'list'>
    print(len(data_set))     # 50000
    print(type(label_set))   # <class 'list'>
    print(len(label_set))    # 50000
    # print_data(data_set, label_set, 10)

    # step 2: training...
    print("\nstep 2: training...")
    kmeans_classify(np.array(data_set), np.array(label_set), 10)
    # kmeans_plus_classify(np.array(data_set), np.array(label_set), 10)


if __name__ == '__main__':
    hand_writing()
    print('OK')
