import numpy as np
import gzip
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
    imgs = np.reshape(imgs, [img_num, width * height])  # reshape为[60000,784]型数组
    return imgs, head


def get_label(buf):  # 得到标签数据
    head = struct.unpack_from('>II', buf, 0)  # 取label文件前2个整形数
    label_num = head[1]
    offset = struct.calcsize('>II')  # 定位到label数据开始的位置
    num_string = '>' + str(label_num) + "B"  # fmt格式：'>60000B'
    labels = struct.unpack_from(num_string, buf, offset)  # 取label数据
    labels = np.reshape(labels, [label_num])  # 转型为列表(一维数组)
    return labels, head


# 用值在0~1间的随机数初始化隶属矩阵U(k行n列)，每列总和为1
def init_fuzzy_mat(n, k):                 # n 个样本，k 个聚类
    fuzzy_mat = np.mat(np.zeros((k, n)))  # k行n列
    for j in range(n):
        col_sum = 0
        randoms = np.random.rand(k-1, 1)  # k-1行1列的“0~1”均匀分布的随机样本值
        # 逐列计算，每列总和为1
        for i in range(k-1):
            fuzzy_mat[i, j] = randoms[i, 0]*(1-col_sum)  # 保证后面不会大于1
            col_sum += fuzzy_mat[i, j]
        fuzzy_mat[-1, j] = 1-col_sum
    print('初始化 fauzzy_mat 成功')
    return fuzzy_mat


# 计算两个向量距离
def cal_distance(v_a, v_b):
    return np.sqrt(np.sum(np.power(v_a - v_b, 2)))


# 计算聚类中心C
def cal_cent_with_fuzzy_mat(data_set, fuzzy_mat, p):
    n, m = data_set.shape
    k = fuzzy_mat.shape[0]
    centroids = np.mat(np.zeros((k, m)))  # 聚类中心，每行表示一个聚类中心
    for i in range(k):
        u_ij = np.power(fuzzy_mat[i, :], p)  # 隶属度的加权指数p
        all_u_ij = np.sum(u_ij)
        numerator = np.array(np.zeros((1, m)))
        for j in range(n):
            numerator += data_set[j]*u_ij[0, j]
        centroids[i, :] = numerator/all_u_ij
    print('计算聚类中心')
    return centroids


# 计算隶属矩阵U
def cal_fuzzy_mat_with_cent(data_set, centroids, p):
    n, m = data_set.shape
    c = centroids.shape[0]
    fuzzy_mat = np.mat(np.zeros((c, n)))
    for i in range(c):
        for j in range(n):
            d_ij = cal_distance(centroids[i, :], data_set[j, :])
            fuzzy_mat[i, j] = 1/np.sum([np.power(d_ij/cal_distance(centroid,
                                        data_set[j, :]), 2/(p-1)) for centroid in centroids])
    print('计算隶属矩阵')
    return fuzzy_mat


# 目标函数：每个类的聚类中心与数据对象的距离平方之和最小
def cal_target_func(data_set, fuzzy_mat, centroids, p):
    n, m = data_set.shape
    c = fuzzy_mat.shape[0]   # ==k
    target_func = 0
    for i in range(c):
        for j in range(n):
            target_func += cal_distance(centroids[i, :], data_set[j, :])**2*np.power(fuzzy_mat[i, j], p)
    return target_func


def fuzzy_c_mean(data_set, k, p):
    n, m = data_set.shape
    fuzzy_mat = init_fuzzy_mat(n, k)

    centroids = cal_cent_with_fuzzy_mat(data_set, fuzzy_mat, p)
    last_target_func = cal_target_func(data_set, fuzzy_mat, centroids, p)
    fuzzy_mat = cal_fuzzy_mat_with_cent(data_set, centroids, p)

    centroids = cal_cent_with_fuzzy_mat(data_set, fuzzy_mat, p)
    target_func = cal_target_func(data_set, fuzzy_mat, centroids, p)
    while last_target_func*0.99 > target_func:
        last_target_func = target_func
        fuzzy_mat = cal_fuzzy_mat_with_cent(data_set, centroids, p)
        centroids = cal_cent_with_fuzzy_mat(data_set, fuzzy_mat, p)
        target_func = cal_target_func(data_set, fuzzy_mat, centroids, p)
    return fuzzy_mat, centroids


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


if __name__ == '__main__':
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
    print(type(data_set))  # <class 'list'>
    print(len(data_set))  # 50000
    print(type(label_set))  # <class 'list'>
    print(len(label_set))  # 50000
    # print_data(data_set, label_set, 10)

    # step 2: training...
    print("\nstep 2: training...")
    k = 10  # 聚类个数
    m = 2   # 隶属度加权指数
    fuzzy_m, cent = fuzzy_c_mean(np.array(data_set), k, m)
    print(fuzzy_m.shape)    # (10, 50000)
    print(type(fuzzy_m))    # <class 'numpy.matrixlib.defmatrix.matrix'>
    # print(fuzzy_m[0])
    # print(np.sum(fuzzy_m, axis=0))  # 每列相加总概率等于1
    print(cent.shape)       # (10, 784)
    print(type(cent))       # <class 'numpy.matrixlib.defmatrix.matrix'>
    # print(cent[0])

    # step 3: 打印效果，将样本最大概率聚类
    fuzzy_list = fuzzy_m.getA().tolist()   # matrix转list
    cluster_index = {}  # 保存 k 个聚类索引号
    for i in range(k):
        cluster_index[i] = []  # 初始化聚类索引
    for j in range(fuzzy_m.shape[1]):
        max_u = 0
        lab = 0
        for i in range(fuzzy_m.shape[0]):
            if fuzzy_list[i][j] > max_u:       # 找出每列中概率最大的
                lab = i
                max_u = fuzzy_list[i][j]
        cluster_index[lab].append(j)        # 将该样本索引聚类

    test_accuracy(cluster_index, label_set)  # 打印聚类效果


