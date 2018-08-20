#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 18:30:56 2018

@author: mazheng
"""
#导入软件包
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
print('导入数据成功')

#数据预处理
print('开始读取数据')
chunks = pd.read_csv('./train_set.csv',chunksize=100,iterator=True)
i = 0
train = []
test = []
for chunk in chunks:
    if i==0:
        chunk = pd.DataFrame(chunk)
        train = chunk.copy()
        i += 1
    else:
        train = train.append(chunk)
print('训练集读取完毕')
chunks = pd.read_csv('./test_set.csv',chunksize=100,iterator=True)
i = 0
for chunk in chunks:
    if i==0:
        chunk = pd.DataFrame(chunk)
        test = chunk.copy()
        i += 1
    else:
        test = test.append(chunk)
print('测试集读取完毕')
del train['article']
del train['id']
del test['article']
print(train.iloc[:,0].size,test.iloc[:,0].size)
print('数据读取完毕')

#特征工程
print('开始构建特征工程')
vectorizer = CountVectorizer(ngram_range=(1,2),min_df=3,max_df=0.9,max_features=100000) #初始化一个CountVectorizer对象
vectorizer.fit(train['word_seg'])#构建词汇表
x_train = vectorizer.transform(train['word_seg'])
x_test = vectorizer.transform(test['word_seg'])
y_train = train['class']-1
print('特征工程构建完毕')

#逻辑回归模型
print('开始训练模型')
lg = LogisticRegression(C=4,dual=True)
lg.fit(x_train,y_train)
print('训练结束')

#保存到本地
test['class'] = y_test.tolist()
test['class'] = test['class']+1
result = test.loc[:,['id','class']]
result.to_csv('./result.csv',index=False)
print('结果保存完成')