#!/usr/bin/env python
# encoding: utf-8
# @author: zxding
# email: d.z.x@qq.com


import numpy as np


def batch_index(length, batch_size, n_iter=100):
    index = range(length)
    for j in xrange(n_iter):
        np.random.shuffle(index)
        for i in xrange(int(length / batch_size)):
            yield index[i * batch_size:(i + 1) * batch_size]


def load_w2v(w2v_file, embedding_dim, debug=False):
    fp = open(w2v_file)
    fp.readline()
    
    w2v = []
    word_dict = dict()
    # [0,0,...,0] represent absent words
    w2v.append([0.] * embedding_dim)
    cnt = 0
    print 'loading word_embedding {}...'.format(w2v_file)
    for line in fp:
        cnt += 1
        line = line.split()
        if len(line) != embedding_dim + 1:
            print 'a bad word embedding: {}'.format(line[0])
            continue
        w2v.append([float(v) for v in line[1:]])
        word_dict[line[0]] = cnt
    print 'done!'
    w2v = np.asarray(w2v, dtype=np.float32)
    #w2v -= np.mean(w2v, axis = 0) # zero-center
    #w2v /= np.std(w2v, axis = 0)
    if debug:
        print 'shape of w2v:',np.shape(w2v)
        print 'id of \'the\':',word_dict['the']
        print 'vector of \'the\':',w2v[word_dict['the']]
    return word_dict, w2v


def change_y_to_onehot(y, n_class = 5):
    onehot = []
    for label in y:
        tmp = [0] * n_class
        tmp[label-1] = 1
        onehot.append(tmp)
    return np.asarray(onehot, dtype=np.int32)


def load_data_for_HN_CNN(input_file, word_to_id, max_sen_len, max_doc_len, n_class=5, encoding='utf8'):
    x, y, sen_len, doc_len = [], [], [], []
    print 'loading input {}...'.format(input_file)
    for line in open(input_file):
        line = line.lower().decode('utf8', 'ignore').split('\t\t')

        t_sen_len = [0] * max_doc_len
        t_x = np.zeros((max_doc_len, max_sen_len), dtype=np.int)
        doc = ' '.join(line[1:])
        sentences = doc.split('<sssss>')
        i = 0
        for sentence in sentences:
            j = 0
            for word in sentence.split():
                if j < max_sen_len:
                    if word in word_to_id:
                        t_x[i, j] = word_to_id[word]
                        j += 1
                else:
                    break
            if j>=3 :
                t_sen_len[i] = j
                i += 1
                if i >= max_doc_len:
                    break
        if i>0 :
            doc_len.append(i)
            sen_len.append(t_sen_len)
            x.append(t_x)
            y.append(int(line[0]))

    y = change_y_to_onehot(y, n_class)
    print 'done!'

    return np.asarray(x), np.asarray(y), np.asarray(sen_len), np.asarray(doc_len)
    