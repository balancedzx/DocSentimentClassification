#!/usr/bin/env python
# encoding: utf-8
# @author: zxding
# email: d.z.x@qq.com


import sys
import time
import os
import numpy as np
import tensorflow as tf
from PrepareData import batch_index, load_charId, load_data_for_CharCNN


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 200, 'number of example per batch')
tf.app.flags.DEFINE_float('learning_rate', 0.0005, 'learning rate')
tf.app.flags.DEFINE_float('keep_prob1', 0.5, 'softmax layer dropout keep prob')
tf.app.flags.DEFINE_float('keep_prob2', 0.5, 'softmax layer dropout keep prob')
tf.app.flags.DEFINE_float('l2_reg', 0.0005, 'l2 regularization')

tf.app.flags.DEFINE_integer('display_step', 1, 'number of test display step')
tf.app.flags.DEFINE_integer('training_iter', 60, 'number of train iter')
tf.app.flags.DEFINE_integer('embedding_dim', 68, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('n_class', 5, 'number of distinct class')
tf.app.flags.DEFINE_integer('max_doc_len', 1014, 'max number of tokens per sentence')

path1 = ['../data/Yelp/min0.1_yelp-2013-train.txt.ss',
         '../data/Yelp/min1.0_yelp-2013-test.txt.ss']
path2 = ['../data/Yelp/min0.2_yelp-2013-train.txt.ss',
         '../data/Yelp/min1.0_yelp-2013-test.txt.ss']
path3 = ['../data/Yelp/min0.5_yelp-2013-train.txt.ss',
         '../data/Yelp/min1.0_yelp-2013-test.txt.ss']
path4 = ['../data/Yelp/min1.0_yelp-2013-train.txt.ss',
         '../data/Yelp/min1.0_yelp-2013-test.txt.ss']
path5 = ['../data/Yelp/min1.0_yelp-2015-train.txt.ss',
         '../data/Yelp/min1.0_yelp-2015-test.txt.ss']
path6 = ['../data/dbpedia/min0.1_train.csv',
          '../data/dbpedia/min1.0_test.csv']
path7 = ['../data/dbpedia/min1.0_train.csv',
          '../data/dbpedia/min1.0_test.csv']
path = path4

tf.app.flags.DEFINE_string('train_file_path', path[0], 'training file')
tf.app.flags.DEFINE_string('test_file_path', path[1], 'testing file')


class CharCNN(object):

    def __init__(self,
                 batch_size=FLAGS.batch_size,
                 learning_rate=FLAGS.learning_rate,
                 keep_prob1=FLAGS.keep_prob1,
                 keep_prob2=FLAGS.keep_prob2,
                 l2_reg=FLAGS.l2_reg,
                 display_step=FLAGS.display_step,
                 training_iter=FLAGS.training_iter,
                 embedding_dim=FLAGS.embedding_dim,
                 n_class=FLAGS.n_class,
                 max_doc_len=FLAGS.max_doc_len,
                 train_file_path=FLAGS.train_file_path,
                 test_file_path=FLAGS.test_file_path
                 ):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.Keep_Prob1 = keep_prob1
        self.Keep_Prob2 = keep_prob2
        self.l2_reg = l2_reg

        self.display_step = display_step
        self.training_iter = training_iter
        self.embedding_dim = embedding_dim
        self.n_class = n_class
        self.max_doc_len = max_doc_len

        self.train_file_path = train_file_path
        self.test_file_path = test_file_path

        self.char_id_mapping, self.c2v = load_charId()
        self.char_embedding = tf.constant(
            self.c2v, dtype=tf.float32, name='char_embedding')

        with tf.name_scope('inputs'):
            self.x = tf.placeholder(tf.int32, [None, self.max_doc_len])
            self.y = tf.placeholder(tf.float32, [None, self.n_class])
            self.doc_len = tf.placeholder(tf.int32, None)
            self.keep_prob1 = tf.placeholder(tf.float32)
            self.keep_prob2 = tf.placeholder(tf.float32)

        def init_variable(shape):
            initial = tf.random_uniform(shape, -0.1, 0.1)
            return tf.Variable(initial)

        with tf.name_scope('weights'):
            self.weights = {
                'conv1': init_variable([7, self.embedding_dim, 1, self.embedding_dim]),
                'conv2': init_variable([7, self.embedding_dim, 1, self.embedding_dim]),
                'conv3': init_variable([3, self.embedding_dim, 1, self.embedding_dim]),
                'conv4': init_variable([3, self.embedding_dim, 1, self.embedding_dim]),
                'conv5': init_variable([3, self.embedding_dim, 1, self.embedding_dim]),
                'conv6': init_variable([3, self.embedding_dim, 1, self.embedding_dim]),
                'linear1': init_variable([34 * self.embedding_dim, 2048]),
                'linear2': init_variable([2048, 2048]),
                'softmax': init_variable([2048, self.n_class])
            }

        with tf.name_scope('biases'):
            self.biases = {
                'conv1': init_variable([self.embedding_dim]),
                'conv2': init_variable([self.embedding_dim]),
                'conv3': init_variable([self.embedding_dim]),
                'conv4': init_variable([self.embedding_dim]),
                'conv5': init_variable([self.embedding_dim]),
                'conv6': init_variable([self.embedding_dim]),
                'linear1': init_variable([2048]),
                'linear2': init_variable([2048]),
                'softmax': init_variable([self.n_class])
            }

    def model(self, inputs):

        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

        def max_pool_3x1(x):
            return tf.nn.max_pool(x, ksize=[1, 3, 1, 1], strides=[1, 3, 1, 1], padding='VALID')

        def AcFun(x):
            return tf.nn.relu(x)

        inputs = tf.reshape(inputs, [-1, self.max_doc_len, self.embedding_dim, 1])

        with tf.name_scope('conv'):
            conv1 = conv2d(inputs, self.weights[
                           'conv1']) + self.biases['conv1']
            conv1 = AcFun(conv1)
            pool1 = max_pool_3x1(conv1)
            output1 = tf.reshape(pool1, [-1, 336, self.embedding_dim, 1])

            conv2 = conv2d(output1, self.weights[
                           'conv2']) + self.biases['conv2']
            conv2 = AcFun(conv2)
            pool2 = max_pool_3x1(conv2)
            output2 = tf.reshape(pool2, [-1, 110, self.embedding_dim, 1])

            conv3 = conv2d(output2, self.weights[
                           'conv3']) + self.biases['conv3']
            conv3 = AcFun(conv3)
            output3 = tf.reshape(conv3, [-1, 108, self.embedding_dim, 1])

            conv4 = conv2d(output3, self.weights[
                           'conv4']) + self.biases['conv4']
            conv4 = AcFun(conv4)
            output4 = tf.reshape(conv4, [-1, 106, self.embedding_dim, 1])

            conv5 = conv2d(output4, self.weights[
                           'conv5']) + self.biases['conv5']
            conv5 = AcFun(conv5)
            output5 = tf.reshape(conv5, [-1, 104, self.embedding_dim, 1])

            conv6 = conv2d(output5, self.weights[
                           'conv6']) + self.biases['conv6']
            conv6 = AcFun(conv6)
            pool6 = max_pool_3x1(conv6)
            output6 = tf.reshape(pool6, [-1, 34 * self.embedding_dim])

        with tf.name_scope('linear'):
            output7 = tf.matmul(output6, self.weights[
                                'linear1']) + self.biases['linear1']
            output7 = AcFun(output7)
            output7 = tf.nn.dropout(output7, keep_prob=self.keep_prob1)

            output8 = tf.matmul(output7, self.weights[
                                'linear2']) + self.biases['linear2']
            output8 = AcFun(output8)
            output8 = tf.nn.dropout(output8, keep_prob=self.keep_prob2)

        with tf.name_scope('softmax'):
            predict = tf.matmul(output8, self.weights[
                                'softmax']) + self.biases['softmax']
            predict = tf.nn.softmax(predict)

        return predict

    def run(self):
        inputs = tf.nn.embedding_lookup(self.char_embedding, self.x)
        prob = self.model(inputs)

        with tf.name_scope('loss'):
            cost = - tf.reduce_mean(self.y * tf.log(prob))
            reg, variables = 0., ['conv1', 'conv2', 'conv3', 'conv4',
                                  'conv5', 'conv6', 'linear1', 'linear2', 'softmax']
            for vari in variables:
                reg += tf.nn.l2_loss(self.weights[vari]) + \
                    tf.nn.l2_loss(self.biases[vari])
            cost += reg * self.l2_reg

        with tf.name_scope('train'):
            global_step = tf.Variable(
                0, name="tr_global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate).minimize(cost, global_step=global_step)

        with tf.name_scope('predict'):
            correct_pred = tf.equal(tf.argmax(prob, 1), tf.argmax(self.y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            correct_num = tf.reduce_sum(tf.cast(correct_pred, tf.int32))

        with tf.name_scope('summary'):
            localtime = time.strftime("%X %Y-%m-%d", time.localtime())
            Summary_dir = 'Summary/' + localtime

            info = 'batch-{}, lr-{}, kb1-{}, kb2-{}, l2_reg-{}'.format(
                self.batch_size,  self.learning_rate, self.Keep_Prob1, self.Keep_Prob2, self.l2_reg)
            info = info + '\n' + self.train_file_path + '\n' + \
                self.test_file_path + '\n' + 'Method: charCNN'
            summary_acc = tf.scalar_summary('ACC ' + info, accuracy)
            summary_loss = tf.scalar_summary('LOSS ' + info, cost)
            summary_op = tf.merge_summary([summary_loss, summary_acc])

            test_acc = tf.placeholder(tf.float32)
            test_loss = tf.placeholder(tf.float32)
            summary_test_acc = tf.scalar_summary('ACC ' + info, test_acc)
            summary_test_loss = tf.scalar_summary('LOSS ' + info, test_loss)
            summary_test = tf.merge_summary(
                [summary_test_loss, summary_test_acc])

            train_summary_writer = tf.train.SummaryWriter(
                Summary_dir + '/train')
            test_summary_writer = tf.train.SummaryWriter(Summary_dir + '/test')

        with tf.name_scope('saveModel'):
            saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
            save_dir = 'Models/' + localtime + '/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

        with tf.name_scope('readData'):
            print '\n----------{}----------'.format(time.strftime("%Y-%m-%d %X", time.localtime()))
            tr_x, tr_y, tr_doc_len = load_data_for_CharCNN(
                self.train_file_path,
                self.char_id_mapping,
                self.max_doc_len,
                self.n_class
            )
            te_x, te_y, te_doc_len = load_data_for_CharCNN(
                self.test_file_path,
                self.char_id_mapping,
                self.max_doc_len,
                self.n_class
            )
            print 'train docs: {}    test docs: {}'.format(len(tr_y), len(te_y))
            print 'training_iter:', self.training_iter
            print info
            print '\n----------{}----------'.format(time.strftime("%Y-%m-%d %X", time.localtime()))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            max_acc, bestIter = 0., 0

            for i in xrange(self.training_iter):

                for train, _ in self.get_batch_data(tr_x, tr_y, tr_doc_len, self.batch_size, self.Keep_Prob1, self.Keep_Prob2):
                    _, step, summary, loss, acc = sess.run(
                        [optimizer, global_step, summary_op, cost, accuracy], feed_dict=train)
                    train_summary_writer.add_summary(summary, step)
                    print 'Iter {}: mini-batch loss={:.6f}, acc={:.6f}'.format(step, loss, acc)
                #saver.save(sess, save_dir, global_step=step)

                if i % self.display_step == 0:
                    acc, loss, cnt = 0., 0., 0
                    for test, num in self.get_batch_data(te_x, te_y, te_doc_len, 2000, keep_prob1=1.0, keep_prob2=1.0):
                        _loss, _acc = sess.run(
                            [cost, correct_num], feed_dict=test)
                        acc += _acc
                        loss += _loss * num
                        cnt += num
                    loss = loss / cnt
                    acc = acc / cnt
                    if acc > max_acc:
                        max_acc = acc
                        bestIter = step

                    summary = sess.run(summary_test, feed_dict={
                                       test_loss: loss, test_acc: acc})
                    test_summary_writer.add_summary(summary, step)
                    print '----------{}----------'.format(time.strftime("%Y-%m-%d %X", time.localtime()))
                    print 'Iter {}: test loss={:.6f}, test acc={:.6f}'.format(step, loss, acc)
                    print 'round {}: max_acc={} BestIter={}\n'.format(i, max_acc, bestIter)

            print 'Optimization Finished!'

    def get_batch_data(self, x, y, doc_len, batch_size, keep_prob1, keep_prob2):
        for index in batch_index(len(y), batch_size, 1):
            feed_dict = {
                self.x: x[index],
                self.y: y[index],
                self.doc_len: doc_len[index],
                self.keep_prob1: keep_prob1,
                self.keep_prob2: keep_prob2,
            }
            yield feed_dict, len(index)


def main(_):
    
    sys.stdout = open('path4_3_01', 'w')
    obj = CharCNN(
    )
    obj.run()

    sys.stdout = open('path5_3_01', 'w')
    obj = CharCNN(
        batch_size=500,
        learning_rate=0.0002,
        n_class=5,
        train_file_path=path5[0],
        test_file_path=path5[1],
    )
    obj.run()
    
    sys.stdout = open('path7_3_01','w')
    obj = CharCNN(
        n_class=14,
        train_file_path=path7[0],
        test_file_path=path7[1],
    )
    obj.run()


if __name__ == '__main__':
    tf.app.run()
