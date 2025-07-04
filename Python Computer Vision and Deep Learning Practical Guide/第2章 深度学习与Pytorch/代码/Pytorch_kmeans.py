import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# 选择设备
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

# 随机矩阵
n_clusters = 4
# 生成数据集
data = make_blobs(n_samples=1000, n_features=2, centers=n_clusters)
matrix = torch.from_numpy(data[0]).to(device).float()
target = data[1]
# 创建KMEANS类
class KMEANS:
    def __init__(
        self, n_clusters=n_clusters, max_iter=None, verbose=False, show=True
    ):
        """
        n_clusters: int 聚类中心数量
        max_iter: int 最大迭代次数
        verbose: bool 是否显示聚类进度
        show: bool 是否展示聚类结果
        """
        self.n_clusters = n_clusters
        # 数据点标签
        self.labels = None
        # 数据之间的距离矩阵
        self.dists = None  # shape: [x.shape[0],n_cluster]
        # 聚类中心点
        self.centers = None
        # 两次聚类距离之间的差值
        self.variation = torch.Tensor([float("Inf")]).to(device)
        self.verbose = verbose
        self.started = False
        self.max_iter = max_iter
        self.count = 0
        self.show = show

    def fit(self, x):
        # 从x中随机选择n_clusters个样本作为初始的聚类中心
        self.plus(x)
        while True:
            # 聚类标记
            self.nearest_center(x)
            # 更新中心点
            self.update_center(x)
            if self.verbose:
                print(self.variation, torch.argmin(self.dists, (0)))
            if torch.abs(self.variation) < 1e-3 and self.max_iter is None:
                break
            elif self.max_iter is not None and self.count == self.max_iter:
                break
            self.count += 1
        if self.show:
            self.show_result(x)

	# 寻找离各数据点最近的中心点，打上标签
    def nearest_center(self, x):
        labels = torch.empty((x.shape[0],)).long().to(device)
        dists = torch.empty((0, self.n_clusters)).to(device)
        # 计算聚类和最近中心点
        for i, sample in enumerate(x):
            dist = torch.sum(
                torch.mul(sample - self.centers, sample - self.centers), (1)
            )
            labels[i] = torch.argmin(dist)
            dists = torch.cat([dists, dist.unsqueeze(0)], (0))
        self.labels = labels
        if self.started:
            self.variation = torch.sum(self.dists - dists)
        self.dists = dists
        self.started = True

	# 更新聚类中心
    def update_center(self, x):
        centers = torch.empty((0, x.shape[1])).to(device)
        for i in range(self.n_clusters):
            # 选出当前聚类中的所有点
            mask = self.labels == i
            cluster_samples = x[mask]
            centers = torch.cat(
                [centers, torch.mean(cluster_samples, (0)).unsqueeze(0)], (0)
            )
        self.centers = centers

    # 展示聚类结果
    def show_result(self, x):
        markers = ["o", "s", "v", "p"]
        if x.shape[1] != 2 or len(set(self.labels.numpy())) > 4:
            raise Exception("只能展示二维数据的聚合结果！")
        print("len", len(set(list(self.labels))))
        for i, label in enumerate(set(list(self.labels.numpy()))):
            samples = x[self.labels == label]
            # print([s[0].item() for s in samples])
            plt.scatter(
                [s[0].item() for s in samples],
                [s[1].item() for s in samples],
                marker=markers[i],
            )
        plt.show()

	# Kmeans ++ 聚类中心初始化
    def plus(self, x):
        num_samples = x.shape[0]
        dim = x.shape[1:]

        # 随机选择一个中心点
        init_row = torch.randint(0, x.shape[0], (1,)).to(device)
        init_points = x[init_row]
        self.centers = init_points

        for i in range(self.n_clusters - 1):
            distances = []
            for row in x:
                # 纪录下所有点到当前所有的中心点的最短距离
                distances.append(
                    torch.min(torch.norm(row - self.centers, dim=1))
                )
            # 蒙特卡洛选取下一个点，距离越长越容易被选择到
            temp = torch.sum(torch.Tensor(distances)) * torch.rand(1)
            for j in range(num_samples):
                temp -= distances[j]
                if temp < 0:
                    self.centers = torch.cat(
                        [self.centers, x[j].unsqueeze(0)], dim=0
                    )
                    break

if __name__ == "__main__":
    import torch.nn as nn
    import time

    a = time.time()
    k = KMEANS(verbose=False)
    k.fit(matrix)
    b = time.time()
    print("total time:{}s ,speed:{}iter/s".format(b - a, (b - a) / k.count))

    # 不同聚类使用不同的形状绘制
    markers = ["o", "s", "v", "p"]
    clf = KMeans(n_clusters=n_clusters)
    clf.fit(matrix)
    for i, label in enumerate(set(clf.labels_)):
        samples = matrix.numpy()[clf.labels_ == label]
        plt.scatter(
            [s[0].item() for s in samples],
            [s[1].item() for s in samples],
            marker=markers[i],
        )
    plt.show()
