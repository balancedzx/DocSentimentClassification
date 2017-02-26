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


def load_charId(embedding_dim=68, debug=False):
    chardic=['a','b','c','d','e','f','g','h','i','j',
    'k','l','m','n','o','p','q','r','s','t','u','v',
    'w','x','y','z','0','1','2','3','4','5','6','7','8','9',
    ',',';','.','!','?',':','\'','\"','/','\\','|','_',
    '@','#','$','%','^','&','*','+','-','=','<','>','(',
    ')','[',']','{','}','`','~']
    char_dict = dict()
    # [0,0,...,0] represent absent words
    w2v = np.array([[0.] * embedding_dim] * (embedding_dim+1))
    #w2v = np.zeros( [embedding_dim+1, embedding_dim] )
    for i in range(embedding_dim):
        w2v[i+1][i] = 1.0
        char_dict[chardic[i]] = i+1
        
    if debug:
        print 'shape of w2v:',np.shape(w2v)
        print 'id of \'d\':',char_dict['d']
        print 'vector of \'d\':',w2v[char_dict['d']]
        while True:
            ch=raw_input('input char')
            print char_dict[ch[0]]
    return char_dict, w2v


def change_y_to_onehot(y, n_class = 5):
    onehot = []
    for label in y:
        tmp = [0] * n_class
        tmp[label-1] = 1
        onehot.append(tmp)
    return np.asarray(onehot, dtype=np.int32)


def load_data_for_CharCNN(input_file, char_to_id, max_doc_len=1014, n_class=5, encoding='utf8'):
    x, y, doc_len = [], [], []
    print 'loading input {}...'.format(input_file)
    for line in open(input_file):
        line = line.lower().decode('utf8', 'ignore').split('\t\t')
        y.append(int(line[0]))

        t_x = np.zeros((max_doc_len), dtype=np.int)
        doc = ' '.join(line[1:])
        sentences = doc.split('<sssss>')
        sentences = ' '.join(sentences)
        i = 0
        for ch in sentences:
            if ch in char_to_id:
                        t_x[i] = char_to_id[ch]
            i+=1
            if i >= max_doc_len:
                break
        doc_len.append(i)
        x.append(t_x)

    y = change_y_to_onehot(y, n_class)
    print 'done!'

    return np.asarray(x), np.asarray(y), np.asarray(doc_len)
    