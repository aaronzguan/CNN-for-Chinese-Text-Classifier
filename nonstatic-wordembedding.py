#coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from numpy import *
import collections
import os
import sys
import tensorflow as tf
from numpy import zeros
from numpy import int32


def _read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        if sys.version_info[0] >= 3:
            return f.read().replace("\n", "<eos>").split('/')
        else:
            return f.read().decode("utf-8").replace("\n", "<eos>").split('/')


# 函数对字典对象，先按value(频数)降序，频数相同的单词再按key(单词)升序。函数返回的是字典对象，key为单词，value为对应的唯一的编号
def _build_vocab(filename, vocab_size):
    data = _read_words(filename)
    # Counter类的目的是用来跟踪值出现的次数。它是一个无序的容器类型，以字典的键值对形式存储，其中元素作为key，其计数作为value。计数值可以是任意的Interger（包括0和负数）
    counter = collections.Counter(data)
    # sort函数和sorted函数唯一的不同是，sort是在容器内排序，sorted生成一个新的排好序的容器
    # sorted(iterable, cmp=None, key=None, reverse=False) --> new sorted list
    # iterable：待排序的可迭代类型的容器;
    # cmp：用于比较的函数，比较什么由key决定,有默认值，迭代集合中的一项;
    # key：用列表元素的某个已命名的属性或函数（只有一个参数并且返回一个用于排序的值）作为关键字，有默认值，迭代集合中的一项;
    # reverse：排序规则. reverse = True 或者 reverse = False，有默认值。
    # 返回值：是一个经过排序的可迭代类型，与iterable一样。
    # 这里key中去负值为了倒叙，由高到低
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    count_pairs = count_pairs[:vocab_size-1]
    # zip函数接受任意多个（包括0个和1个）序列作为参数，返回一个tuple列表
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    return word_to_id


# 把单词文件转化为单词id
def _file_to_word_ids(filename, word_to_id):
    file = open(filename, 'r', encoding='utf-8')
    lines = list(file.readlines())
    data = []
    vocab_size = len(word_to_id)
    for i in range(len(lines)):
        line = lines[i].split('/')
        result = zeros(len(line), int32)
        for i in range(len(line)):
            if line[i] in word_to_id:
                result[i] = word_to_id[line[i]]
            else:
                result[i] = vocab_size-1
        data.append(result)
    return data


# 读取原始训练和测试数据转化为id
def get_data(data_path=None, vocab_size=30000):
    train_input_path = os.path.join(data_path, "trainData_input.txt")
    train_label_path = os.path.join(data_path, "trainData_label.txt")
    test_input_path = os.path.join(data_path, "testData_input.txt")
    test_label_path = os.path.join(data_path, "testData_label.txt")

    word_to_id = _build_vocab(train_input_path, vocab_size)
    train_input_data = _file_to_word_ids(train_input_path, word_to_id)
    file = open(train_label_path, 'r', encoding='utf-8')
    train_label_data = list(file.readlines())
    file.close()
    test_input_data = _file_to_word_ids(test_input_path, word_to_id)
    file = open(test_label_path, 'r', encoding='utf-8')
    test_label_data = list(file.readlines())
    file.close()
    vocabulary = len(word_to_id)
    return train_input_data, train_label_data, test_input_data, test_label_data, vocabulary


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)  # 变量的初始值为截断正太分布
    return tf.Variable(initial, name='W')


# 初始化偏置常量的函数
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name='b')


def add_conv_layer(config,kernel_height,kernel_num,input):
    with tf.name_scope("conv"):
        # Kernel = 64 of 5(h) x 200(w)
        W_conv = weight_variable([kernel_height, config.vector_len, 1, kernel_num])  # 卷积核大小，输入通道数目，输出通道数目
        tf.summary.histogram("W_conv", W_conv)
        # Create bias for each output: total 64
        b_conv = bias_variable([kernel_num])
        # Convolution
        output_conv = tf.nn.conv2d(input, W_conv, strides=[1, 1, 1, 1], padding='VALID', name='conv')
        # Activate Function ELU
        output_conv1_elu = tf.nn.elu(output_conv + b_conv)
        return tf.reduce_max(output_conv1_elu, 1)  # Use reduce_max to record the largest number from each feature map
# reduce_max(output_conv1_elu, 1). Here the 1 is 'axis'
# 'axis' indicating which dimension of the reduce operation will be applied on
# axis = 0: column, axis = 1: row
# For example
# a = array([[0, 1], [2, 3]])
# reduce_max(a, 0) results [2,3]
# reduce_max(a, 1) results [1,3]


def run():
    config = Config()

    with tf.name_scope("input_layer"):
        x_ = tf.placeholder(tf.int32, [None], name="x_")

    with tf.name_scope("embedding_layer"):
        with tf.device("/cpu:0"):
            # 词向量矩阵数组
            embedding = tf.get_variable("embedding", [config.vocab_size, config.vector_len], dtype=tf.float32)
            tf.summary.histogram("embedding", embedding)
            # 将词转化为词向量
            input_embedding = tf.nn.embedding_lookup(embedding, x_)
            input_vector = tf.reshape(input_embedding, [1, -1, config.vector_len, 1])

    with tf.name_scope("conv_layer"):
        output_convs = []
        for i in range(len(config.kernel)):
            # 有多少个种卷积核就循环多少次
            # 每次循环把每种卷积核的所有reduced_max_feature连锁起来形成 max_pool_layer
            output_convs.append(add_conv_layer(config, config.kernel[i][0], config.kernel[i][1], input_vector))
            # Concatenate the features generated from each kind of kernel
            output_max_pool = tf.concat(output_convs, 2)  # Dim = [batch_size, in_height, in_width, in_channels]; 2 is width
        tf.summary.histogram("output_max_pool", output_max_pool)

    with tf.name_scope("full_connect_layer"):
        kernel_num = 0
        # Calculate the total number of kernel, which is the total number of features input into full connect layer
        for i in range(len(config.kernel)):
            kernel_num = kernel_num + config.kernel[i][1]
        W_fc1 = weight_variable([kernel_num, config.fc_num])
        tf.summary.histogram("W_fc1", W_fc1)
        b_fc1 = bias_variable([config.fc_num])
        # Reshape the input [1, kernel_num]
        input_fc1 = tf.reshape(output_max_pool, [1, kernel_num])
        h_fc1 = tf.nn.elu(tf.matmul(input_fc1, W_fc1) + b_fc1)  # matmul: one-to-one multiply for 2 vectors
        # Perform dropout to avoid over-fitting
        output_fc1_drop = tf.nn.dropout(h_fc1, config.keep_prob)

    # The final softmax layer then receives this feature vector as input and uses it to classify the sentence
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
        # cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_output + 1e-10), reduction_indices=[1]))
        # cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_output ), reduction_indices=[1]))
        tf.summary.scalar("cross_entropy", cross_entropy)

    with tf.name_scope("train"):
        # Adam Optimizer is used to train the parameter
        train_step = tf.train.AdamOptimizer(config.learn_rate).minimize(cross_entropy)

    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(y_output, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.histogram("accuracy", accuracy)

    mylog = open(config.log_path, 'a', encoding='utf-8')  # 'a' = append: append the log
    print("不预先训练词向量方式")
    mylog.write("不预先训练词向量方式" + "\n")
    print("迭代次数： ", config.epoch_num)
    mylog.write("迭代次数： " + str(config.epoch_num) + "\n")
    mylog.flush()  # Call this method to make sure that all pending events have been written to disk.
    print("向量维度： ", config.vector_len)
    mylog.write("向量维度： " + str(config.vector_len) + "\n")
    mylog.flush()
    print("词典规模： ", config.vocab_size)
    mylog.write("词典规模： " + str(config.vocab_size) + "\n")
    mylog.flush()
    print("卷积类型： ", config.conv_padding)
    mylog.write("卷积类型： " + config.conv_padding + "\n")
    mylog.flush()
    print("卷积核种类： ", str(len(config.kernel)))
    mylog.write("卷积核种类： " + str(len(config.kernel)) + "\n")
    mylog.flush()
    for i in range(len(config.kernel)):
        print("卷积核： ", i, "  高度： ", config.kernel[i][0], "  数量： ", config.kernel[i][1])
        mylog.write(
            "卷积核： " + str(i) + "  高度： " + str(config.kernel[i][0]) + "  数量： " + str(config.kernel[i][1]) + "\n")
        mylog.flush()

    print("全连接层神经元数量： ", config.fc_num)
    mylog.write("全连接层神经元数量： " + str(config.fc_num) + "\n")
    mylog.flush()
    print("全连接层dropout：  ", config.keep_prob)
    mylog.write("全连接层dropout：  " + str(config.keep_prob) + "\n")
    mylog.flush()

    sess = tf.InteractiveSession()
    summary_merge = tf.summary.merge_all()
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:  # tensorflow version < 0.12
        writer = tf.train.SummaryWriter(config.summary_path, sess.graph)
        init = tf.initialize_all_variables()
    else:  # tensorflow version >= 0.12
        writer = tf.summary.FileWriter(config.summary_path, sess.graph)
        init = tf.global_variables_initializer()

    sess.run(init)
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

    train_input_data, train_label_data, test_input_data, test_label_data, vocabulary \
        = get_data(config.data_path, config.vocab_size)
    max_kernel_height = 0
    for i in range(len(config.kernel)):
        if config.kernel[i][1] > max_kernel_height:
            max_kernel_height = config.kernel[i][1]
    for i in range(config.epoch_num):

        train_accuracy = 0.0

        for j in range(len(train_input_data)):
            train_input_line = train_input_data[j]
            if len(train_input_line) <= max_kernel_height:
                continue
            train_output_line = zeros([1, config.class_num], int32)  # the output is the [1,28] of all 0s
            index = int(train_label_data[j])
            train_output_line[0, index-1] = 1  # Set 1 at the index of the label
            train_step.run(feed_dict={x_: train_input_line, y_: train_output_line})
            train_accuracy = train_accuracy + accuracy.eval(feed_dict={x_: train_input_line, y_: train_output_line})
            if j % config.print_interval == 0:
                trac = train_accuracy/(j+1)
                print("第 %d次迭代，第 %d 轮，训练平均准确率为： %g" % (i+1, j, trac))
                mystr = "第 %d次迭代，第 %d 轮，训练平均准确率为： %g\n" % (i+1, j, trac)
                mylog.write(mystr)
                mylog.flush()
                result_summary = sess.run(summary_merge, feed_dict={x_: train_input_line, y_: train_output_line})
                writer.add_summary(result_summary, j)

        final_accuracy = 0

        for j in range(len(test_input_data)):
            test_input_line = test_input_data[j]
            if len(test_input_line) <= max_kernel_height:
                continue
            test_output_line = zeros([1, config.class_num], int32)  # the output is the [1,28] of all 0s
            index = int(test_label_data[j])
            test_output_line[0, index - 1] = 1
            final_accuracy = final_accuracy + accuracy.eval(feed_dict={x_: test_input_line, y_: test_output_line})

        print("第 %d 次迭代，测试平均准确率： %g" % (i + 1, final_accuracy / len(test_input_data)))
        mylog.write("第 %d 次迭代，测试平均准确率： %g" % (i + 1, final_accuracy / len(test_input_data)))
        mylog.flush()


class Config(object):
    vector_len = 200  # 词向量的长度 = Kernel的 宽度
    epoch_num = 13  # 整个文本循环次数
    keep_prob = 0.5  # 用于dropout，每批数据输入时神经网络中的每个单元会以1-keep_prob的概率不工作，可以防止过拟合
    vocab_size = 30000  # 词典规模，总共10K个词
    conv_padding = "VALID"  # 定义卷积的类型；SAME是等卷积
    kernel = [[5, 64],
              # [4, 48],
              # [3, 32],
              # [2, 16],
              # [1, 8],
              ]
    # kernel = [height, num], the width is the length of word vector, which is 200
    fc_num = 300  # 定义隐藏层神经元数量
    class_num = 28  # 定义类别数量
    learn_rate = 0.0001  # 定义学习率
    print_interval = 3000  # 每隔多少轮训练输出一次结果

    data_path = "/Users/apple/Desktop/NLP/textclassfier/text/data_fenci/finalData/"
    log_path = "/Users/apple/Desktop/NLP/textclassfier/log/log.txt"
    summary_path = "/Users/apple/Desktop/NLP/textclassfier/summary/"
    word2vec_model_path = "/Users/apple/Desktop/NLP/textclassfier/text/data_fenci/finalData/word2vec_model"


run()


