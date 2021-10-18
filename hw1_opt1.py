# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 14:10:01 2021

@author: Brilliant Coder
"""
import os
import re
import numpy as np
import math
import time

file_path = './199801_clear.txt'
stop_words_path = './baidu_stopwords.txt'

# 加载文档
with open(file_path,'r',encoding='gbk') as f:
    ori_txt = f.read()

# 加载停用词表
with open(stop_words_path,'r',encoding='utf-8') as f:
    stop_words = f.read().strip().splitlines()

# 划分文档，提取词汇，过滤停用词
ori_doc_list = ori_txt.split('\n\n')
doc_list = [] # 存储文档的单词及其词频
all_words_dict = {} # 存储所有不重复单词
word_index = 0
regx = re.compile(r'/.\w*')

preprocess_start = time.time()
for doc in ori_doc_list:
    sentences = doc.split('\n')
    doc_words = {}
    for sentence in sentences:
        words = sentence.split()
        words = words[1:]
        for word in words:
            if '/w' in word:
                continue
            word = regx.sub("", word)
            if word not in stop_words and word!='':
                if doc_words.get(word)==None:
                    doc_words.update({word:1})
                else:
                    doc_words[word]+=1
                
                if all_words_dict.get(word)==None:
                    all_words_dict.update({word:word_index})
                    word_index+=1
                    
    doc_list.append(doc_words)
    
preprocess_end = time.time()
print("文档数量:",len(doc_list))
print("所有不重复词的数量：",len(all_words_dict))
print("文本预处理耗时: %fs" % (preprocess_end-preprocess_start))

# 计算idf
cal_idf_st = time.time()
idf_dict={}
for word in all_words_dict.keys():
    num = 0
    for doc in doc_list:
        if doc.get(word):
            num+=1    
    idf_dict[word]=num
print("计算idf耗时：%fs" % (time.time()-cal_idf_st))

# 将文档转化为向量表示，同时单位化
vectorized_st = time.time()
doc_vectors = np.zeros(shape=(len(doc_list),len(all_words_dict)),dtype=np.float32())
n = len(doc_list)
for i in range(n):
    for word,tf in doc_list[i].items():
        doc_vectors[i][all_words_dict[word]] = tf * (math.log2(n/idf_dict[word]) + 1)
    norm = np.linalg.norm(doc_vectors[i])
    # if norm==0.0:
    #     print(doc_vectors[i],i)
    if norm!=0.0:  # 0/0会有警告,然后向量赋值为nan,最终余弦相似度也会为nan
        doc_vectors[i] = doc_vectors[i]/norm

print("文档向量化并单位化耗时：%fs" % (time.time()-vectorized_st))

# 计算文档两两之间的相似度。单位化之后计算相似度就是两个向量内积
def cosine_similarity(vector1,vector2):
    inner_product = np.sum(vector1 * vector2)
    
    return inner_product

sim_st = time.time()
sim_matrix = []
for i in range(n):
    sim_list = []
    for j in range(i+1,n):
        sim_list.append(cosine_similarity(doc_vectors[i], doc_vectors[j]))
    sim_matrix.append(sim_list)
    #print("文档 %d与其后面的相似度计算完成" % i)
    
print("优化余弦相似度后计算相似度耗时：%fs" % (time.time()-sim_st))

with open('./similarity_opt1_res.txt','w') as f:
    for i in range(len(sim_matrix)):
        f.write(str(sim_matrix[i])+'\n')
