#!/usr/bin/env python
# encoding: utf-8
# @author: zxding
# email: d.z.x@qq.com


import sys
import time
import os
import numpy as np
import tensorflow as tf
from PrepareData import batch_index, load_w2v, load_data_for_HN_CNN


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 2000, 'number of example per batch')
tf.app.flags.DEFINE_float('learning_rate', 0.005, 'learning rate')
tf.app.flags.DEFINE_float('keep_prob1', 1.0, 'word embedding training dropout keep prob')
tf.app.flags.DEFINE_float('keep_prob2', 1.0, 'softmax layer dropout keep prob')
tf.app.flags.DEFINE_float('l2_reg', 0.0000, 'l2 regularization')

tf.app.flags.DEFINE_integer('display_step', 1, 'number of test display step')
tf.app.flags.DEFINE_integer('training_iter', 40, 'number of train iter')
tf.app.flags.DEFINE_integer('embedding_dim', 200, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('n_class', 5, 'number of distinct class')
tf.app.flags.DEFINE_integer('max_doc_len', 15, 'max number of tokens per sentence')
tf.app.flags.DEFINE_integer('max_sentence_len', 20,'max number of tokens per sentence')
tf.app.flags.DEFINE_integer('n_hidden', 50, 'number of hidden unit')

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
tf.app.flags.DEFINE_string('embedding_file_path', '../data/Yelp/yelp-2013-vectors.txt', 'embedding file')


class HN_GatedCNN_biGRNN(object):

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
                 max_sentence_len=FLAGS.max_sentence_len,
                 n_hidden=FLAGS.n_hidden,
                 train_file_path=FLAGS.train_file_path,
                 test_file_path=FLAGS.test_file_path,
                 w2v_file=FLAGS.embedding_file_path,
                 embedding_type=0,
                 scope='sentence',
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
        self.max_sentence_len = max_sentence_len
        self.n_hidden = n_hidden

        self.train_file_path = train_file_path
        self.test_file_path = test_file_path
        self.w2v_file = w2v_file
        self.scope = scope

        self.word_id_mapping, self.w2v = load_w2v(
            self.w2v_file, self.embedding_dim)
        if embedding_type == 0:  # Pretrained and Untrainable
            self.word_embedding = tf.constant(
                self.w2v, dtype=tf.float32, name='word_embedding')
        elif embedding_type == 1:  # Pretrained and Trainable
            self.word_embedding = tf.Variable(
                self.w2v, dtype=tf.float32, name='word_embedding')
        elif embedding_type == 2:  # Random and Trainable
            self.word_embedding = tf.Variable(tf.random_uniform(
                [len(self.word_id_mapping) + 1, self.embedding_dim], -0.1, 0.1), name='word_embedding')

        with tf.name_scope('inputs'):
            self.x = tf.placeholder(
                tf.int32, [None, self.max_doc_len, self.max_sentence_len])
            self.y = tf.placeholder(tf.float32, [None, self.n_class])
            self.sen_len = tf.placeholder(tf.int32, [None, self.max_doc_len])
            self.doc_len = tf.placeholder(tf.int32, None)
            self.keep_prob1 = tf.placeholder(tf.float32)
            self.keep_prob2 = tf.placeholder(tf.float32)

        def init_variable(shape):
            initial = tf.random_uniform(shape, -0.01, 0.01)
            return tf.Variable(initial)

        with tf.name_scope('weights'):
            self.weights = {
                'conv1': init_variable([3, self.embedding_dim, 1, self.n_hidden]),
                'conv2': init_variable([2, self.embedding_dim, 1, self.n_hidden]),
                'conv3': init_variable([1, self.embedding_dim, 1, self.n_hidden]),
                'gconv1': init_variable([3, self.embedding_dim, 1, self.n_hidden]),
                'gconv2': init_variable([2, self.embedding_dim, 1, self.n_hidden]),
                'gconv3': init_variable([1, self.embedding_dim, 1, self.n_hidden]),
                'softmax': init_variable([2 * self.n_hidden, self.n_class]),
            }
            self.cell_fw = tf.nn.rnn_cell.GRUCell(self.n_hidden)
            self.cell_bw = tf.nn.rnn_cell.GRUCell(self.n_hidden)

        with tf.name_scope('biases'):
            self.biases = {
                'conv1': init_variable([self.n_hidden]),
                'conv2': init_variable([self.n_hidden]),
                'conv3': init_variable([self.n_hidden]),
                'gconv1': init_variable([self.n_hidden]),
                'gconv2': init_variable([self.n_hidden]),
                'gconv3': init_variable([self.n_hidden]),
                'softmax': init_variable([self.n_class]),
            }

    def model(self, inputs):

        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

        def max_pool_3x1(x):
            return tf.nn.max_pool(x, ksize=[1, 3, 1, 1], strides=[1, 3, 1, 1], padding='VALID')

        def AcFun(x):
            return tf.nn.relu(x)

        inputs = tf.nn.dropout(inputs, keep_prob=self.keep_prob1)
        inputs = tf.reshape(
            inputs, [-1, self.max_sentence_len, self.embedding_dim, 1])

        with tf.name_scope('word_encode'):
            conv1 = conv2d(inputs, self.weights[
                           'conv1']) + self.biases['conv1']
            gconv1 = conv2d(inputs, self.weights[
                            'gconv1']) + self.biases['gconv1']
            conv1 = conv1 * tf.nn.sigmoid(gconv1)
            h_conv1 = AcFun(conv1)
            outputs1 = tf.reshape(
                h_conv1, [-1, self.max_sentence_len - 2, self.n_hidden])

            conv2 = conv2d(inputs, self.weights[
                           'conv2']) + self.biases['conv2']
            gconv2 = conv2d(inputs, self.weights[
                            'gconv2']) + self.biases['gconv2']
            conv2 = conv2 * tf.nn.sigmoid(gconv2)
            h_conv2 = AcFun(conv2)
            outputs2 = tf.reshape(
                h_conv2, [-1, self.max_sentence_len - 1, self.n_hidden])

            conv3 = conv2d(inputs, self.weights[
                           'conv3']) + self.biases['conv3']
            gconv3 = conv2d(inputs, self.weights[
                            'gconv3']) + self.biases['gconv3']
            conv3 = conv3 * tf.nn.sigmoid(gconv3)
            h_conv3 = AcFun(conv3)
            outputs3 = tf.reshape(
                h_conv3, [-1, self.max_sentence_len - 0, self.n_hidden])

        with tf.name_scope('word_attention'):
            outputs1 = self.reduce_mean(
                outputs1, tf.maximum(self.sen_len - 2, 0))
            outputs2 = self.reduce_mean(
                outputs2, tf.maximum(self.sen_len - 1, 0))
            outputs3 = self.reduce_mean(
                outputs3, tf.maximum(self.sen_len - 0, 0))
            outputs = (outputs1 + outputs2 + outputs3) / 3.0

        outputs = tf.reshape(outputs, [-1, self.max_doc_len, self.n_hidden])
        with tf.name_scope('sentence_encode'):
            outputs, state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=self.cell_fw,
                cell_bw=self.cell_bw,
                inputs=outputs,
                sequence_length=self.doc_len,
                dtype=tf.float32,
                scope=self.scope
            )
            outputs = tf.concat(2, outputs)  # batch_size * doc_len * 2n_hidden

        with tf.name_scope('sentence_attention'):
            outputs = self.reduce_mean(outputs, self.doc_len)

        with tf.name_scope('softmax'):
            outputs = tf.nn.dropout(outputs, keep_prob=self.keep_prob2)
            predict = tf.matmul(outputs, self.weights[
                                'softmax']) + self.biases['softmax']
            predict = tf.nn.softmax(predict)

        return predict

    def reduce_mean(self, inputs, length):
        length = tf.reshape(length, [-1])
        maskshape = [tf.shape(inputs)[0], tf.shape(inputs)[1], 1]
        mask = tf.reshape(tf.cast(tf.sequence_mask(
            length, tf.shape(inputs)[1]), tf.float32), maskshape)
        inputs *= mask

        length = tf.cast(tf.reshape(length, [-1, 1]), tf.float32) + 1e-9
        inputs = tf.reduce_sum(inputs, 1, keep_dims=False) / length
        return inputs

    def run(self):
        inputs = tf.nn.embedding_lookup(self.word_embedding, self.x)
        prob = self.model(inputs)

        with tf.name_scope('loss'):
            cost = - tf.reduce_mean(self.y * tf.log(prob))
            reg, variables = tf.nn.l2_loss(self.word_embedding), [
                'conv1', 'conv2', 'conv3', 'gconv1', 'gconv2', 'gconv3', 'softmax']
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
                self.test_file_path + '\n' + 'Method: HN_GatedCNN-biGRNN'
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
            print '----------{}----------'.format(time.strftime("%Y-%m-%d %X", time.localtime()))
            tr_x, tr_y, tr_sen_len, tr_doc_len = load_data_for_HN_CNN(
                self.train_file_path,
                self.word_id_mapping,
                self.max_sentence_len,
                self.max_doc_len,
                n_class=self.n_class
            )
            te_x, te_y, te_sen_len, te_doc_len = load_data_for_HN_CNN(
                self.test_file_path,
                self.word_id_mapping,
                self.max_sentence_len,
                self.max_doc_len,
                n_class=self.n_class
            )
            print 'train docs: {}    test docs: {}'.format(len(tr_y), len(te_y))
            print 'training_iter:', self.training_iter
            print info
            print '----------{}----------'.format(time.strftime("%Y-%m-%d %X", time.localtime()))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            max_acc, bestIter = 0., 0

            for i in xrange(self.training_iter):

                for train, _ in self.get_batch_data(tr_x, tr_y, tr_sen_len, tr_doc_len, self.batch_size, self.Keep_Prob1, self.Keep_Prob2):
                    _, step, summary, loss, acc = sess.run(
                        [optimizer, global_step, summary_op, cost, accuracy], feed_dict=train)
                    train_summary_writer.add_summary(summary, step)
                    print 'Iter {}: mini-batch loss={:.6f}, acc={:.6f}'.format(step, loss, acc)
                #saver.save(sess, save_dir, global_step=step)

                if i % self.display_step == 0:
                    acc, loss, cnt = 0., 0., 0
                    for test, num in self.get_batch_data(te_x, te_y, te_sen_len, te_doc_len, 2000, keep_prob1=1.0, keep_prob2=1.0):
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

    def get_batch_data(self, x, y, sen_len, doc_len, batch_size, keep_prob1, keep_prob2):
        for index in batch_index(len(y), batch_size, 1):
            feed_dict = {
                self.x: x[index],
                self.y: y[index],
                self.sen_len: sen_len[index],
                self.doc_len: doc_len[index],
                self.keep_prob1: keep_prob1,
                self.keep_prob2: keep_prob2,
            }
            yield feed_dict, len(index)


def main(_):
    logfile='path4_0_01'
    sys.stdout = open(logfile,'w')
    obj = HN_GatedCNN_biGRNN(
                scope=logfile
    )
    obj.run()

    logfile='path5_0_01'
    sys.stdout = open(logfile,'w')
    obj = HN_GatedCNN_biGRNN(
                n_class=5,
                train_file_path=path5[0],
                test_file_path=path5[1],
                scope=logfile,
    )
    obj.run()

    logfile = 'path7_0_01'
    sys.stdout = open(logfile, 'w')
    obj = HN_GatedCNN_biGRNN(
        n_class=14,
        train_file_path=path7[0],
        test_file_path=path7[1],
        scope=logfile,
    )
    obj.run()


if __name__ == '__main__':
    tf.app.run()
