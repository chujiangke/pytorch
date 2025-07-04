import pickle
import os.path as osp
import numpy as np

from config import cifar_folder


class Cifar:
    def __init__(self, folder=cifar_folder):
        self.folder = folder
        self.files = [osp.join(self.folder, "data_batch_%d" % n) for n in range(1, 6)]

    def load_pickle(self, path):
        f = open(path, "rb")
        data_dict = pickle.load(f, encoding="bytes")
        X = data_dict[b"data"]
        Y = data_dict[b"labels"]
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)  # .astype("float")
        Y = np.array(Y)
        return X, Y

    def load_cifar10(self):
        xs = []
        ys = []
        for file in self.files:
            X, Y = self.load_pickle(file)
            xs.append(X)
            ys.append(Y)
        train_x = np.concatenate(xs)
        train_y = np.concatenate(ys)

        test_x, test_y = self.load_pickle(osp.join(self.folder, "test_batch"))
        return train_x, train_y, test_x, test_y


if __name__ == "__main__":
    data = Cifar()
    train_x, train_y, test_x, test_y = data.load_cifar10()
    print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

