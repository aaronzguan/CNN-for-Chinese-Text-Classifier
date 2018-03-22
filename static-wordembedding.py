#coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from numpy import *
import collections
import os
import sys
import string
import tensorflow as tf
from numpy import zeros
from numpy import int32
from gensim.models import word2vec
from tensorflow.python import debug as tf_debug


def _read_lines(filename):
    with tf.gfile.GFile(filename, "r") as f:
        if sys.version_info[0] >= 3:
            return f.readlines()
        else:
            return f.readlines().decode("utf-8")


# 读取原始训练和测试数据转化为id
def get_data(data_path=None):
    train_input_path = os.path.join(data_path, "trainData_input.txt")
    train_label_path = os.path.join(data_path, "trainData_label.txt")
    test_input_path = os.path.join(data_path, "testData_input.txt")
    test_label_path = os.path.join(data_path, "testData_label.txt")

    train_input_data = _read_lines(train_input_path)

    file = open(train_label_path, 'r', encoding='utf-8')
    train_label_data = list(file.readlines())
    file.close()

    test_input_data = _read_lines(test_input_path)

    file = open(test_label_path, 'r', encoding='utf-8')
    test_label_data = list(file.readlines())
    file.close()

    return train_input_data, train_label_data, test_input_data, test_label_data


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)  # 变量的初始值为截断正太分布
    return tf.Variable(initial, name='W')


# 初始化偏置常量的函数
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name='b')


def add_conv_layer(config, kernel_height, kernel_num, input):
    with tf.name_scope("conv"):
        W_conv = weight_variable(
            [kernel_height, config.vector_len, 1, kernel_num])  # 卷积核大小，输入通道数目，输出通道数目
        tf.summary.histogram("W_conv", W_conv)
        b_conv = bias_variable([kernel_num])
        output_conv = tf.nn.conv2d(input, W_conv, strides=[1, 1, 1, 1], padding='VALID', name='conv')
        output_conv1_elu = tf.nn.elu(output_conv + b_conv)
        return tf.reduce_max(output_conv1_elu, 1)


def run():
    config = Config()

    with tf.name_scope("input_layer"):
        x_ = tf.placeholder(tf.float32, [None, 200], name="x_")  # [?,200]
        input_vector = tf.reshape(x_, [1, -1, config.vector_len, 1])  # [1,?,200,1] [batch_size, in_height, in_width, in_channels]
    with tf.name_scope("conv_layer"):
        output_convs = []
        for i in range(len(config.kernel)):
            output_convs.append(add_conv_layer(config, config.kernel[i][0], config.kernel[i][1], input_vector))
            output_max_pool = tf.concat(output_convs, 2)  # Dim = [batch_size, in_height, in_width, in_channels]; 2 is width
        tf.summary.histogram("output_max_pool", output_max_pool)

    with tf.name_scope("full_connect_layer"):
        kernel_num = 0  # Total Kernel Numbers, which is also the total no. of input of fully connect layer
        for i in range(len(config.kernel)):
            kernel_num = kernel_num+config.kernel[i][1]
        W_fc1 = weight_variable([kernel_num, config.fc_num])
        tf.summary.histogram("W_fc1", W_fc1)
        b_fc1 = bias_variable([config.fc_num])
        input_fc1 = tf.reshape(output_max_pool, [1, kernel_num])
        h_fc1 = tf.nn.elu(tf.matmul(input_fc1, W_fc1) + b_fc1)
        output_fc1_drop = tf.nn.dropout(h_fc1, config.keep_prob)

    with tf.name_scope("softmax"):
        W_sfm = weight_variable([config.fc_num, config.class_num])
        tf.summary.histogram("W_sfm", W_sfm)
        b_sfm = bias_variable([config.class_num])
        y_output = tf.nn.softmax(tf.matmul(output_fc1_drop, W_sfm) + b_sfm)  # [1，28]
        tf.summary.histogram("y_output", y_output)

    with tf.name_scope("label"):
        y_ = tf.placeholder(tf.float32, [None])
        y_ = tf.reshape(y_, [1, config.class_num])

    with tf.name_scope("loss"):
        # 防止y_output出现值为0的情况，不然tf.log(0)会报NAN
        x = tf.zeros_like(y_output)
        y__ = tf.equal(y_output, x)
        y__ = tf.cast(y__, dtype=tf.float32)
        y__ = tf.multiply(y__, 1e-10)
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_output + y__), reduction_indices=[1]))
        # cross_entropy = -tf.reduce_mean(tf.square(tf.subtract(y_ ,y_output)))
        tf.summary.scalar("cross_entropy", cross_entropy)

    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(config.learn_rate).minimize(cross_entropy)

    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(y_output, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.histogram("accuracy", accuracy)

    mylog = open(config.log_path, 'a', encoding='utf-8')
    print("预先训练词向量方式")
    mylog.write("预先训练词向量方式" + "\n")
    print("迭代次数： ", config.epoch_num)
    mylog.write("迭代次数： "+str(config.epoch_num) + "\n")
    mylog.flush()
    print("向量维度： ", config.vector_len)
    mylog.write("向量维度： "+str(config.vector_len) + "\n")
    mylog.flush()
    print("词典规模： ", config.vocab_size)
    mylog.write("词典规模： "+str(config.vocab_size) + "\n")
    mylog.flush()
    print("卷积类型： ", config.conv_padding)
    mylog.write("卷积类型： "+config.conv_padding + "\n")
    mylog.flush()
    print("卷积核数量： ", str(len(config.kernel)))
    mylog.write("卷积核种类： "+ str(len(config.kernel)) + "\n")
    mylog.flush()
    for i in range(len(config.kernel)):
        print("卷积核： ", i, "  高度： ", config.kernel[i][0], "  数量： ", config.kernel[i][1])
        mylog.write("卷积核： " + str(i) + "  高度： "+str(config.kernel[i][0]) + "  数量： " + str(config.kernel[i][1]) + "\n")
        mylog.flush()
    print("全连接层神经元数量： ", config.fc_num)
    mylog.write("全连接层神经元数量： " + str(config.fc_num)+ "\n")
    mylog.flush()
    print("全连接层dropout：  ", config.keep_prob)
    mylog.write("全连接层dropout：  " + str(config.keep_prob)+ "\n")
    mylog.flush()

    sess = tf.InteractiveSession()
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
    summary_merge = tf.summary.merge_all()
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:  # tensorflow version < 0.12
        writer = tf.train.SummaryWriter(config.summary_path, sess.graph)
        init = tf.initialize_all_variables()
    else:  # tensorflow version >= 0.12
        writer = tf.summary.FileWriter(config.summary_path, sess.graph)
        init = tf.global_variables_initializer()

    sess.run(init)

    train_input_data, train_label_data, test_input_data, test_label_data = get_data(config.data_path)
    word2vec_model = word2vec.Word2Vec.load(config.word2vec_model_path)
    dic = word2vec_model.wv
    max_kernel_height = 0
    for i in range(len(config.kernel)):
        if config.kernel[i][1] > max_kernel_height:
            max_kernel_height = config.kernel[i][1]
    for i in range(config.epoch_num):
        train_accuracy = 0.0
        for j in range(len(train_input_data)):
            train_input_line = train_input_data[j].split('/')
            if len(train_input_line) <= max_kernel_height:
                continue
            for k in range(len(train_input_line)):
                if not train_input_line[k] in dic:
                    train_input_line[k] = "unknown"
            train_input_vector = word2vec_model.wv[train_input_line]
            train_output_line = zeros([1, config.class_num], int32)
            index = int(train_label_data[j])
            train_output_line[0, index-1] = 1
            train_step.run(feed_dict={x_: train_input_vector, y_: train_output_line})
            train_accuracy = train_accuracy+accuracy.eval(feed_dict={x_: train_input_vector, y_: train_output_line})
            if j % config.print_interval == 0:
                trac = train_accuracy/(j+1)
                print("第 %d次迭代，第 %d 轮，训练平均准确率为： %g" % (i+1, j, trac))
                mystr = "第 %d次迭代，第 %d 轮，训练平均准确率为： %g\n" % (i+1, j, trac)
                mylog.write(mystr)
                mylog.flush()
                result_summary = sess.run(summary_merge, feed_dict={x_: train_input_vector, y_: train_output_line})
                writer.add_summary(result_summary, j)

        final_accuracy = 0
        for j in range(len(test_input_data)):
            test_output_line = test_input_data[j].split('/')
            if len(test_output_line) <= max_kernel_height:
                continue
            for k in range(len(test_output_line)):
                if not test_output_line[k] in dic:
                    test_output_line[k] = "unknown"
            test_input_vector = word2vec_model.wv[test_output_line]
            test_output_line = zeros([1, config.class_num], int32)
            index = int(test_label_data[j])
            test_output_line[0, index - 1] = 1
            final_accuracy = final_accuracy + accuracy.eval(feed_dict={x_: test_input_vector, y_: test_output_line})

        print("第 %d 次迭代，测试平均准确率： %g" % (i + 1, final_accuracy / len(test_input_data)))
        mylog.write("第 %d 次迭代，测试平均准确率： %g" % (i + 1, final_accuracy / len(test_input_data)))
        mylog.flush()




class Config(object):
    epoch_num = 13  # 整个文本循环次数
    vector_len = 200#词向量的长度
    vocab_size = 30000  # 词典规模，总共10K个词
    conv_padding = "VALID"#定义卷积的类型SAME是等卷积
    kernel = [[5, 64],
              # [4, 48],
              # [3, 32],
              # [2, 16],
              # [1, 8],
              ]
    fc_num = 300  # 定义隐藏层神经元数量
    keep_prob = 0.5  # 用于dropout，每批数据输入时神经网络中的每个单元会以1-keep_prob的概率不工作，可以防止过拟合
    class_num = 28  # 定义类别数量
    learn_rate = 0.001  # 定义学习率
    print_interval = 3000  # 每隔多少轮训练输出一次结果

    data_path = "/Users/apple/Desktop/NLP/textclassfier/text/data_fenci/finalData/"
    log_path = "/Users/apple/Desktop/NLP/textclassfier/log/log.txt"
    summary_path = "/Users/apple/Desktop/NLP/textclassfier/summary/"
    word2vec_model_path = "/Users/apple/Desktop/NLP/textclassfier/text/data_fenci/finalData/word2vec_model"


run()


