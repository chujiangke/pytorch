from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


def classification(use_tf=True):
    # 获取训练集
    train_news = fetch_20newsgroups(subset="train")
    x_train = train_news.data
    y_train = train_news.target

    # 获取验证集合
    test_news = fetch_20newsgroups(subset="test")
    x_test = test_news.data
    y_test = test_news.target

    if use_tf:
        # Tf-idf统计
        vec = TfidfVectorizer()
    else:
        # 词频统计
        vec = CountVectorizer()
    # 转化为向量
    x_count_train = vec.fit_transform(x_train)
    x_count_test = vec.transform(x_test)
    # 定义贝叶斯分类模型
    clf = MultinomialNB()
    # 训练模型
    clf.fit(x_count_train, y_train)
    y_pred = clf.predict(x_count_test)

    # 计算准确率
    print(
        "Using tf-idf : {} Accuracy: {} ".format(
            use_tf, sum(y_pred == y_test) / len(y_test)
        )
    )


if __name__ == "__main__":
    classification(use_tf=False)
    classification()
