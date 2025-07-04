from glob import glob
import numpy as np 
from sklearn.svm import SVC,SVR,LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from tqdm import tqdm
import os

feature_paths = sorted(glob(r"D:\datasets\EnglishHnd\English\Hnd\Img\*\*.npy"))
labels = [os.path.split(os.path.dirname(im))[-1] for im in feature_paths]

label_set = sorted(list(set(labels)))
label_dict = dict(zip(label_set,[i for i in range(len(label_set))]))

label_ids = [label_dict[label] for label in labels]

features = []
for feature_path in tqdm(feature_paths):
    feature = np.load(feature_path)
    features.append(feature)

features = np.array(features)

x_train,x_test,y_train,y_test = train_test_split(features,label_ids,test_size=0.15)


print("fitting")
# clf = OneVsRestClassifier(SVC())
clf = LinearSVC(multi_class="ovr",verbose=True,max_iter=10000)

clf.fit(x_train,y_train)

yp = clf.predict(x_test)

print(np.sum(yp == y_test)/len(y_test))

import matplotlib.pyplot as plt 

plt.scatter(range(len(y_train)),y_train,color = 'b')
plt.scatter(range(len(yp)),y_test,color = 'r')
plt.scatter(range(len(yp)),yp,color = 'g')
plt.show()


# 最佳准确率73%