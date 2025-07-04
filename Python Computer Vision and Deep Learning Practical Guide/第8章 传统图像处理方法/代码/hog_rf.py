from skimage.feature import hog
from load_cifar import Cifar
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


class SVMClassifier:
    def __init__(self):
        self.data = Cifar()
        self.train_x, self.train_y, self.test_x, self.test_y = (
            self.data.load_cifar10()
        )
        
        self.clf = RandomForestClassifier(
            n_estimators=800, min_samples_leaf=5, verbose=True, n_jobs=-1
        )
       
        print("loading train data")
        self.train_hog = []
        for img in tqdm(self.train_x):
            self.train_hog.append(self.extract_feature(img))
        print("loading test data")
        self.test_hog = []
        for img in tqdm(self.test_x):
            self.test_hog.append(self.extract_feature(img))

    def extract_feature(self, img):
        hog_feat = hog(
            img,
            orientations=9,
            pixels_per_cell=[3, 3],
            cells_per_block=[2, 2],
            feature_vector=True,
        )
        # print(hog_feat.shape)
        return hog_feat

    def fit(self):
        self.clf.fit(self.train_hog, self.train_y)

    def evaluate(self):

        train_pred = self.clf.predict(self.train_hog)
        train_accuracy = sum(train_pred == self.train_y) / len(self.train_y)
        print("train accuracy : {}".format(train_accuracy))

        test_pred = self.clf.predict(self.test_hog)
        test_accuracy = sum(test_pred == self.test_y) / len(self.test_y)
        print("test accuracy : {}".format(test_accuracy))


if __name__ == "__main__":
    clf = SVMClassifier()
    clf.fit()
    clf.evaluate()
