#encoding=utf-8
import jieba
import os
import sys
import os.path
import  re
import string
import fileinput

rootdir = "D:/python/textclassfier/text/newdata/"
jieba.load_userdict("/Users/apple/Desktop/NLP/textclassfier/dic/default.dic")  # "D:/python/textclassfier/dic/default.dic"

stopword = [line.strip() for line in open('/Users/apple/Desktop/NLP/textclassfier/stopword/stopwords.txt','r',encoding= 'utf-8').readlines()]  # D:/python/textclassfier/stopword/stopwords.txt
stopword = set(stopword)
n = 0;
for parent, dirnames, filenames in os.walk(rootdir):    #三个参数：分别返回1.父目录 2.所有文件夹名字（不含路径） 3.所有文件名字
    for dirname in dirnames:                       #输出文件夹信息
        print("parent is:" + parent)
        print("dirname is:" + dirname)
    for filename in filenames:                        #输出文件信息

         print("the full name of the file is:" + os.path.join(parent, filename)) #输出文件路径信息
         fileOutput = open("D:/python/textclassfier/text/data_fenci/"+filename, "w", encoding='utf-8')
         file = open(os.path.join(parent,filename),'r',encoding= 'utf-8')
         while 1:
             line = file.readline()
             if not line:
                 break
             #print(line)
             words = list(jieba.cut(line,cut_all=False))
             deleteNum=0;
             #print(len(words))
             for item in range(len(words)):
                 # re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", words[item-deleteNum])  # 中文标点
                 #words[item - deleteNum]=words[item-deleteNum].translate(string.punctuation)
                 if(words[item-deleteNum] in stopword or len(re.sub("[\u4e00-\u9fa5]+", "", words[item - deleteNum]))>0):
                     del words[item-deleteNum]
                     deleteNum=deleteNum+1
             # print(str(n)+" "+"/".join(words))
             fileOutput.write(str(n)+" "+"/".join(words)+"\n")
         n=n+1;
         file.close()
         fileOutput.close()
         print(os.path.join(parent,filename)+"   文件已处理完，共处理了"+str(n+1)+"个文件")

         #exit(0)





#jieba.add_word('石墨烯')
# test_sent = (
# "李小福是创新办主任也是云计算方面的专家; 什么是八一双鹿\n"
# "例如我输入一个带“韩玉赏鉴”的标题，在自定义词库中也增加了此词为N类\n"
# "「台中」正確應該不會被切開。mac上可分出「石墨烯」；此時又可以分出來凱特琳了。"
# )
# words = jieba.cut(test_sent)
# print('/'.join(words))
# print("="*40)