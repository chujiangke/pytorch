# path = "D:\\datasets\\dev_urls.txt"
path = "D:\\datasets\\eval_urls.txt"
folder = "D:\\datasets\\pubfig_eval"

import pandas as pd
import os
from urllib.request import urlretrieve
from sklearn.utils import shuffle
from tqdm import tqdm
import socket
from time import ctime, time
import requests

socket.setdefaulttimeout(5)

# 第一行是注释，第二行标题前面有个#号
df = pd.read_table(path, header=1)
cols = df.columns[1:]
df = df.drop(["md5sum"], axis=1)
df.columns = cols


def download(i, df):
    print("thread {} started in {}".format(i, ctime()))
    # 打乱df，便于多进程运行
    df = shuffle(df)
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        url = row["url"]
        target_folder = os.path.join(folder, row["person"])
        if not os.path.exists(target_folder):
            os.mkdir(target_folder)
        target_path = os.path.join(
            target_folder, row["person"] + str(row["imagenum"]) + ".jpg"
        )
        if os.path.exists(target_path):
            continue
        # 先创建空文件，这样下载失败之后，其他进程（或线程）便不会再重复尝试
        with open(target_path, "wb") as f:
            try:
                r = requests.get(url, timeout=3)
                f.write(r.content)
            except Exception as e:
                pass


import threading

threads = []
for i in range(4):
    t = threading.Thread(target=download, args=(i, df))
    threads.append(t)

if __name__ == "__main__":
    MULTI_THREAD = True
    if MULTI_THREAD:
        for t in threads:
            t.start()
        print("Done")
    else:
        download(0, df)

