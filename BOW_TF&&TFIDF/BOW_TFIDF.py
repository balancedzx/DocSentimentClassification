#!/usr/bin/env python
# encoding: utf-8
# @author: zxding
# email: d.z.x@qq.com


import re
import math
import collections
import numpy as np


def batch_index(length, batch_size, n_iter=100):
    index = range(length)
    for j in xrange(n_iter):
        np.random.shuffle(index)
        for i in xrange(int(length / batch_size)):
            yield index[i * batch_size:(i + 1) * batch_size]


def getWordlist(input_file, wordlist='wordlist.txt', num=50000, debug=False):
    words, DFwords = [], []
    doc_len = 0
    for line in open(input_file):
        line = re.split('\t\t|\"', line.lower())
        doc = ' '.join(line[1:])
        doc = doc.split()
        doc_set = list(set(doc))
        if debug:
            print '<doc>: ' + str(doc)
            print '<doc_set>: ' + str(doc_set)

        words.extend(doc)
        DFwords.extend(doc_set)
        doc_len += 1
        if doc_len % 20000 == 0:
            print doc_len

    collect1 = collections.Counter(words).most_common(num)
    collect2 = collections.Counter(DFwords)
    g = open(wordlist, 'w')
    for word, _ in collect1:
        g.write(word + '\t' + str(math.log(doc_len / (collect2[word] + 0.))) + '\n')


def change_y_to_onehot(y, n_class=5):
    onehot = []
    for label in y:
        tmp = [0] * n_class
        tmp[label - 1] = 1
        onehot.append(tmp)
    return np.asarray(onehot, dtype=np.int32)


def getwordIDF(path):
    count, wordId, wordIDF = 0, dict(), dict()
    for line in open(path):
        line = line.split()
        wordId[line[0]] = count
        wordIDF[count] = float(line[1])
        count += 1
    return wordId, wordIDF


def load_data_for_BOW(input_file, wordlist='wordlist.txt', n_class=14, num=50000, useIDF=True):
    wordId, wordIDF = getwordIDF(wordlist)
    x, y = [], []
    print 'loading input {}...'.format(input_file)
    for line in open(input_file):
        line = re.split('\t\t|\"', line.lower())
        doc = ' '.join(line[1:])
        doc = doc.split()

        t_x, wordnum = dict(), 0
        for word in doc:
            if word in wordId:
                wordnum += 1
                Id = wordId[word]
                if Id in t_x:
                    t_x[Id] += 1
                else:
                    t_x[Id] = 1

        for Id in t_x:
            t_x[Id] = t_x[Id] / (wordnum + 0.)
            if useIDF:
                t_x[Id] = t_x[Id] * wordIDF[Id]

        # print 't_x',t_x
        y.append(int(line[0]))
        x.append(t_x)

    y = change_y_to_onehot(y, n_class)
    print 'done!'
    return np.asarray(x), np.asarray(y)


def dict_to_array(batch, num=50000):
    ret = []
    for t_x in batch:
        tmp = np.zeros([num], np.float32)
        for Id in t_x:
            tmp[Id] = t_x[Id]
        ret.append(tmp)
    return np.array(ret)


if __name__ == '__main__':
    getWordlist('../data/dbpedia/min1.0_train.csv', 'wordlist_dbp.txt')
    getWordlist('../data/Yelp/min1.0_yelp-2013-train.txt.ss', 'wordlist_yelp2013.txt')
    getWordlist('../data/Yelp/min1.0_yelp-2015-train.txt.ss', 'wordlist_yelp2015.txt')
