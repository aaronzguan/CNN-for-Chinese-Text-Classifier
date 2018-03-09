#coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import fileinput
import collections
import os
import sys
import tensorflow as tf
from numpy import zeros
from numpy import int32

from gensim.models import word2vec
from gensim.models.keyedvectors import KeyedVectors

# 把文本读取为一个一个的词

sentences=word2vec.Text8Corpus(u"/Users/apple/Desktop/NLP/textclassfier/text/data_fenci/finalData/replaceUnknown.txt")  # D:/python/textclassfier/text/data_fenci/finalData/replaceUnknown.txt
model = word2vec.Word2Vec(sentences, size=200, window=5, workers=5, iter=7, sorted_vocab=1,)
print(model.wv["下车"])
model.save("/Users/apple/Desktop/NLP/textclassfier/text/data_fenci/finalData/word2vec_model")  # D:/python/textclassfier/text/data_fenci/finalData/word2vec_model

print(model.wv["下车","不用","不一样","不知道"])





def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    if sys.version_info[0] >= 3:
      return f.read().replace("\n", "<eos>").split(' ')
    else:
      return f.read().decode("utf-8").replace("\n", "<eos>").split(' ')

#函数对字典对象，先按value(频数)降序，频数相同的单词再按key(单词)升序。函数返回的是字典对象，key为单词，value为对应的唯一的编号
def _build_vocab(filename):
  data = _read_words(filename)
  #Counter类的目的是用来跟踪值出现的次数。它是一个无序的容器类型，以字典的键值对形式存储，其中元素作为key，其计数作为value。计数值可以是任意的Interger（包括0和负数）
  counter = collections.Counter(data)
  #sort函数和sorted函数唯一的不同是，sort是在容器内排序，sorted生成一个新的排好序的容器
  #sorted(iterable, cmp=None, key=None, reverse=False) --> new sorted list
  #iterable：待排序的可迭代类型的容器;
  #cmp：用于比较的函数，比较什么由key决定,有默认值，迭代集合中的一项;
  #key：用列表元素的某个已命名的属性或函数（只有一个参数并且返回一个用于排序的值）作为关键字，有默认值，迭代集合中的一项;
  #reverse：排序规则. reverse = True 或者 reverse = False，有默认值。
  #返回值：是一个经过排序的可迭代类型，与iterable一样。
  #这里key中去负值为了倒叙，由高到低
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
  count_pairs =count_pairs[:30000-1]
  #zip函数接受任意多个（包括0个和1个）序列作为参数，返回一个tuple列表
  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))
  return word_to_id

# 把单词文件转化为单词id
def _file_to_word_ids(filename, word_to_id):
  file=open(filename,'r',encoding= 'utf-8')
  lines=list(file.readlines())
  data=[]
  for i in range(len(lines)):
    line=lines[i].split('/')
    result = zeros(len(line), int32)
    for i in range(len(line)):
      if (line[i] in word_to_id):
        result[i] = word_to_id[line[i]]
      else:
        result[i] = 9999
    data.append(result)
  return data


# 读取原始训练和测试数据转化为id
def get_data(data_path=None):
    train_input_path = os.path.join(data_path, "trainData_input.txt")
    train_label_path = os.path.join(data_path, "trainData_label.txt")
    test_input_path = os.path.join(data_path, "testData_input.txt")
    test_label_path = os.path.join(data_path, "testData_label.txt")

    word_to_id = _build_vocab(train_input_path)
    train_input_data = _file_to_word_ids(train_input_path, word_to_id)
    # train_input_data = tf.convert_to_tensor(train_input_data, name="train_input_data", dtype=tf.int32)

    file = open(train_label_path, 'r', encoding='utf-8')
    train_label_data = list(file.readlines())
    file.close()
    # train_label_data = tf.convert_to_tensor(train_label_data, name="train_label_data", dtype=tf.int32)

    #train_label_data=list(fileinput.input(train_label_path))
    test_input_data = _file_to_word_ids(test_input_path, word_to_id)
    # test_input_data = tf.convert_to_tensor(test_input_data, name="test_input_data", dtype=tf.int32)
    file = open(test_label_path, 'r', encoding='utf-8')
    test_label_data = list(file.readlines())
    file.close()
    # test_label_data = tf.convert_to_tensor(test_label_data, name="test_label_data", dtype=tf.int32)

    #test_label_data = list(fileinput.input(test_label_path))
    vocabulary = len(word_to_id)
    return train_input_data, train_label_data, test_input_data,test_label_data, vocabulary


def replaceUnknown(filename):
    word_to_id= _build_vocab(filename)
    file = open(filename, 'r', encoding='utf-8')
    fileOutput = open("D:/python/textclassfier/text/data_fenci/finalData/replaceUnknown.txt" , "w", encoding='utf-8')
    lines = list(file.readlines())
    for i in range(len(lines)):
        line = lines[i].split(' ')
        for j in range(len(line)):
            if (not (line[j] in word_to_id)):
                line[j]="unknown";
        newline=" ".join(line)
        fileOutput.write(newline + "\n")

# replaceUnknown("D:/python/textclassfier/text/data_fenci/finalData/allData.txt")