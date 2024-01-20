import numpy as np


# 因为总共的评论词是10000个，所以每个评论的词汇总量也不会超过10000，就用10000的array，1表示某个词汇存在，0表示不存在处理
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.
    return results

# 将标签向量化有两种方法：既可以将标签列表转换为一个整数张量，
# 也可以使用one-hot编码。one-hot编码是分类数据的一种常用格式，也叫分类编码（categorical encoding）。
# 在这个例子中，标签的one-hot编码就是将每个标签表示为全零向量，只有标签索引对应的元素为1
def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results


## 关于vectorize_sequences和to_one_hot
# vectorize_sequences是一共有10000个词，同时每条评论也不会超过10000个词，所以方法就是构造一个array，有这个词，那么词的index标记1
# to_one_hot，不同于上面的情况，新闻分类，一共有8982条记录，但每条记录只会是0~45之间的一个数字，看代码就是把一个array，其每个成员的取值范围是0~45
# 变成了一个二维数组，行数是8982，而列数是46，对于每行而言，只有数字的那个index是1，其余位置是0
# 简而言之，这两种方法都能把一个数字变成0，1连个数字组成的序列，vectorize_sequences一行会有多个1，而to_one_hot一行只有1个1