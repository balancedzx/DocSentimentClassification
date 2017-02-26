#!/usr/bin/env python
# encoding: utf-8
# @author: zxding
# email: d.z.x@qq.com


import sys
import time
import os
import numpy as np
import tensorflow as tf
from BOW_TFIDF import batch_index, load_data_for_BOW, dict_to_array


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 2000, 'number of example per batch')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'learning rate')
tf.app.flags.DEFINE_float('l2_reg', 0.0000, 'l2 regularization')

tf.app.flags.DEFINE_integer('display_step', 1, 'number of test display step')
tf.app.flags.DEFINE_integer('training_iter', 10, 'number of train iter')
tf.app.flags.DEFINE_integer('feature_dim', 50000, 'dimension of feature')
tf.app.flags.DEFINE_integer('n_class', 5, 'number of distinct class')
tf.app.flags.DEFINE_boolean('useIDF', True, 'True:BOW-TFIDF  False:BOW-TF')

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

wordlist = ['wordlist_dbp.txt', 'wordlist_yelp2013.txt', 'wordlist_yelp2015.txt']

tf.app.flags.DEFINE_string('train_file_path', path[0], 'training file')
tf.app.flags.DEFINE_string('test_file_path', path[1], 'testing file')
tf.app.flags.DEFINE_string('wordlist', wordlist[1], 'testing file')


class Softmax(object):

    def __init__(self,
                 batch_size=FLAGS.batch_size,
                 learning_rate=FLAGS.learning_rate,
                 l2_reg=FLAGS.l2_reg,
                 display_step=FLAGS.display_step,
                 training_iter=FLAGS.training_iter,
                 feature_dim=FLAGS.feature_dim,
                 n_class=FLAGS.n_class,
                 useIDF=FLAGS.useIDF,
                 train_file_path=FLAGS.train_file_path,
                 test_file_path=FLAGS.test_file_path,
                 wordlist=FLAGS.wordlist
                 ):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg

        self.display_step = display_step
        self.training_iter = training_iter
        self.feature_dim = feature_dim
        self.n_class = n_class
        self.useIDF = useIDF

        self.train_file_path = train_file_path
        self.test_file_path = test_file_path
        self.wordlist = wordlist

        with tf.name_scope('inputs'):
            self.x = tf.placeholder(tf.float32, [None, self.feature_dim])
            self.y = tf.placeholder(tf.float32, [None, self.n_class])

        def init_variable(shape):
            initial = tf.random_uniform(shape, -0.01, 0.01)
            return tf.Variable(initial)

        with tf.name_scope('weights'):
            self.weights = {
                'softmax': init_variable([self.feature_dim, self.n_class])
            }

        with tf.name_scope('biases'):
            self.biases = {
                'softmax': init_variable([self.n_class])
            }

    def model(self, inputs):
        with tf.name_scope('softmax'):
            predict = tf.matmul(inputs, self.weights[
                                'softmax']) + self.biases['softmax']
            predict = tf.nn.softmax(predict)

        return predict

    def run(self):
        inputs = self.x
        prob = self.model(inputs)

        with tf.name_scope('loss'):
            cost = - tf.reduce_mean(self.y * tf.log(prob))
            reg, variables = 0., ['softmax']
            for vari in variables:
                reg += tf.nn.l2_loss(self.weights[vari]) + \
                    tf.nn.l2_loss(self.biases[vari])
            cost += reg * self.l2_reg

        with tf.name_scope('train'):
            global_step = tf.Variable(
                0, name="tr_global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate).minimize(cost, global_step=global_step)
            #optimizer = tf.train.MomentumOptimizer(self.learning_rate,0.9).minimize(cost, global_step=global_step)
            #optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cost, global_step=global_step)

        with tf.name_scope('predict'):
            correct_pred = tf.equal(tf.argmax(prob, 1), tf.argmax(self.y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            correct_num = tf.reduce_sum(tf.cast(correct_pred, tf.int32))

        with tf.name_scope('summary'):
            localtime = time.strftime("%Y-%m-%d %X", time.localtime())
            Summary_dir = 'Summary/' + localtime

            info = 'batch-{}, lr-{}, l2_reg-{}'.format(
                self.batch_size,  self.learning_rate, self.l2_reg)
            method = 'BOW-TF'
            if self.useIDF:
                method = 'BOW-TFIDF'
            info = info + '\n' + self.train_file_path + '\n' + \
                self.test_file_path + '\n' + 'Method: ' + method
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
            print 'wordlist:'+self.wordlist
            tr_x, tr_y = load_data_for_BOW(
                input_file=self.train_file_path,
                wordlist=self.wordlist,
                n_class=self.n_class,
                num=self.feature_dim,
                useIDF=self.useIDF
            )
            te_x, te_y = load_data_for_BOW(
                input_file=self.test_file_path,
                wordlist=self.wordlist,
                n_class=self.n_class,
                num=self.feature_dim,
                useIDF=self.useIDF
            )
            print 'train docs: {}    test docs: {}'.format(len(tr_y), len(te_y))
            print 'training_iter:', self.training_iter
            print info
            print '\n----------{}----------'.format(time.strftime("%Y-%m-%d %X", time.localtime()))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            # saver.restore(sess, 'models/logs/1481624500__r0.005_b2000_l0.0001self.softmax/-280')
            max_acc, bestIter = 0., 0

            for i in xrange(self.training_iter):

                for train, _ in self.get_batch_data(tr_x, tr_y, self.batch_size):
                    _, step, summary, loss, acc = sess.run(
                        [optimizer, global_step, summary_op, cost, accuracy], feed_dict=train)
                    train_summary_writer.add_summary(summary, step)
                    print 'Iter {}: mini-batch loss={:.6f}, acc={:.6f}'.format(step, loss, acc)
                #saver.save(sess, save_dir, global_step=step)

                if i % self.display_step == 0:
                    acc, loss, cnt = 0., 0., 0
                    for test, num in self.get_batch_data(te_x, te_y, 2000):
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
                    print 'round {}: max_acc={:.4f} BestIter={}\n'.format(i, max_acc, bestIter)

            print 'Optimization Finished!'

    def get_batch_data(self, x, y, batch_size):
        for index in batch_index(len(y), batch_size, 1):
            tmp = dict_to_array(x[index], num=self.feature_dim)
            feed_dict = {
                self.x: tmp,
                self.y: y[index],
            }
            yield feed_dict, len(index)


def main(_):
    
    sys.stdout = open('path4_0_01', 'w')
    obj = Softmax(
    )
    obj.run()

    sys.stdout = open('path5_0_01', 'w')
    obj = Softmax(
        n_class=5,
        train_file_path=path5[0],
        test_file_path=path5[1],
        wordlist=wordlist[2],
    )
    obj.run()

    sys.stdout = open('path7_0_01', 'w')
    obj = Softmax(
        batch_size=200,
        training_iter=20,
        n_class=14,
        train_file_path=path7[0],
        test_file_path=path7[1],
        wordlist=wordlist[0],
    )
    obj.run()

    sys.stdout = open('TF_path4_0_01', 'w')
    obj = Softmax(
        training_iter=20,
        useIDF=False
    )
    obj.run()

    sys.stdout = open('TF_path5_0_01', 'w')
    obj = Softmax(
        n_class=5,
        train_file_path=path5[0],
        test_file_path=path5[1],
        wordlist=wordlist[2],
        useIDF=False
    )
    obj.run()

    sys.stdout = open('TF_path7_0_01', 'w')
    obj = Softmax(
        batch_size=200,
        training_iter=20,
        n_class=14,
        train_file_path=path7[0],
        test_file_path=path7[1],
        wordlist=wordlist[0],
        useIDF=False
    )
    obj.run()


if __name__ == '__main__':
    tf.app.run()
