# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 23:50:17 2021

@author: xumic
"""

import numpy as np
import pandas as pd
import csv
import sys
import string
import nltk
#nltk.downloader.download('vader_lexicon')
#nltk.downloader.download('wordnet')
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import wordnet as wn

from scipy.sparse import hstack


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from textblob import TextBlob


pd.set_option('display.max_columns', 100)






# load training data

'''
train_full = pd.read_csv('hw3_train.txt', header=None, sep=" ",delimiter='\t')
train_full.columns= ["labels","review"]
print("full training data")
print(train_full)
'''
train_full = pd.read_csv('train_exclude_test6000.csv',names=["labels","headers","review"])
#train_full_alt.columns= ["labels","headers","review"]

train_full['labels'] = train_full['labels'].apply(lambda x: x - 1)

#train_full_alt=train_full[:10000]
sample_ratio=0.2
train_full_alt = train_full.sample(frac= sample_ratio, random_state=100) 


print(train_full_alt["labels"])
print(train_full_alt['headers'])
print(train_full_alt['review'])

# split training data
train_ratio=0.7
random_seed=100

#train_df = train_full.sample(frac= train_ratio, random_state=100) 




'''
#train_df=train_full.head(20997)
valid_df = train_full.drop(train_df.index)
print('training set size:', len(train_df))
print('validation set size:', len(valid_df))
'''


train_df = train_full_alt.sample(frac= train_ratio, random_state=100) 
valid_df = train_full_alt.drop(train_df.index)


#load test data
test_df_full = pd.read_csv('test.csv', names=["labels","headers","review"])
test_df=test_df_full[:3000]
#test_df.columns= ["id","headers","review"]
#print(test_df)









# sentiment prediction##############################################################










# calculate sentiment score########################################










sentiment=SentimentIntensityAnalyzer()
#sentiment_dict=sentiment.polarity_scores(words)
#return(sentiment_dict['compound'])
'''
#some tests
words="not what I was expecting"
sentiment_dict=sentiment.polarity_scores(words)
print("sentiment test!!!!!!!!!!!!!!!!!!")
print(sentiment_dict['compound'])
'''

# separate review from titles, and create columns
def separate_review_text_from_title(df):
    header = []
    contexts = []
    for index, row in df.iterrows():
        title = row['review']
        position = title.find(':')
        #end_position = title.find('/>')
        contexts.append(title[position+2 :])
        header.append(title[:position] )
    return header, contexts


def make_headers_review_to_lists(df):
    header=[]
    contexts=[]
    for x in df['headers']:
        header.append(x)
    for y in df['review']:
        contexts.append(y)
        
    return header, contexts




# building feature extractor for word vector######################

en_stop = set(nltk.corpus.stopwords.words('english'))


#create corpus by extracting all words in reviews, currently only for review text themselves
def create_corpus(df):
    
    sentence_list=[]
    for x in df['review']:
        sentence_list.append(x)
    
    #remove puncrtuations 
    no_punc_list=[]
    #punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    for y in sentence_list:
        no_punc_list.append(y.translate(str.maketrans('', '', string.punctuation)))
        #print(y)
        #print(y)
        
    #tokenize   
    token_list=[] 
    for z in no_punc_list:
        token_list.append(z.strip().lower().split())
        #print(z)
        
    #merge the lists into a corpus
    corpus=[]
    for x in token_list:
        for word in x:
            corpus.append(word)
            
    #corpus=set(corpus)
    #corpus=list(corpus)
    #remove stop words#####
    corpus=[x for x in corpus if x not in en_stop]
    #print(len(corpus))
    return corpus
   
'''  
   
# prcoessing corpus, pick 2000 most common words, can be commented out after obtaining improved features

full_corpus=create_corpus(train_full_alt)
    
from collections import Counter

counter=Counter(full_corpus)
most_common_K=counter.most_common(4000)
#print(most_common_K)


most_common_list=[] # this variabel can be used for either most important features or simply most common words
for x in most_common_K:
    most_common_list.append(x[0])
  
#abc=TextBlob(train_df['review'][1]).sentiment.polarity#print("seeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee abc")
#print(abc)
'''   
#use the most important k features instead, can be commented out

my_file = open("improved_features.txt", "r")
content = my_file.read()
most_common_list = content.split(",")
most_common_list=[x for x in most_common_list if len(x)>=3]
#adjust feature numbers based importance, can be commented out
most_common_list=most_common_list[:2000]
print(most_common_list)
my_file.close()









#vectorizer = CountVectorizer(stop_words = None).fit(most_common_list)


#create feature df using most frequent words###########
def create_doc_features(df):
    #process the review into word list
    
    temp_list=[]
    for x in df['review']:
        temp_list.append(x)
    
    no_punc_list=[]
    for x in temp_list:
        no_punc_list.append(x.translate(str.maketrans('', '', string.punctuation)))
        
    
    doc_list=[] 
    for z in no_punc_list:
        doc_list.append(z.strip().lower().split())
    
    #create feature df based on  whether the 2000 frequent words appear
    vector_feature_df=pd.DataFrame()   
    
    for wd in most_common_list:
        vector_feature_df['contains({})'.format(wd)]=0
    
    for word in most_common_list:
        row_list=[]#[0]*len(df)
        #print("is it hereererererereeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
        i=0
        #can alternate between appearance counts and true/false
        for d in doc_list:
            row_list.append(word in d)
            #for wd in d:
                #if wd == word:
                    #row_list[i]+=1
            i+=1
        #print("is it hereererererereeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")     
        vector_feature_df['contains({})'.format(word)]=row_list
    #print(vector_feature_df)
        
     
    return vector_feature_df
    '''
    sent_list=df['review'].tolist()
    X=vectorizer.transform(sent_list)
    return X
    '''
#testing first!!!!!!!!!!!!!!!!!!!!!!
#train_alt=create_doc_features(train_df,vectorizer)
#valid_alt=create_doc_features(valid_df,vectorizer)
#test_alt=create_doc_features(test_df,vectorizer)



#create feature df, currently with sentiments, word vectors, and length##############
def create_other_feature_df(df):

    header,text=make_headers_review_to_lists(df)
    train_X=pd.DataFrame()
    #train_X['header']=header
    #train_X['text']=text
    #print("some testssssssssssssssssssss")
    #print(train_df['labels'])
    #print(train_df['review'])
    #print(train_df['header'])
    #print(train_df['text'])
    
    #sentiment_score
    train_X['header_positive_score']=-1000
    train_X['header_neutral_score']=-1000
    train_X['header_negative_score']=-1000
    
    train_X['text_positive_score']=-1000
    train_X['text_neutral_score']=-1000
    train_X['text_negative_score']=-1000
    #print(train_X)
    #print(train_df['review'])
    
    #sentiments need additional work   ##################
    i=0
    for x in header:
        sentiment_dict=sentiment.polarity_scores(str(x))
        train_X.at[i,'header_positive_score']=sentiment_dict['pos']
        train_X.at[i,'header_neutral_score']=sentiment_dict['neu']
        train_X.at[i,'header_negative_score']=sentiment_dict['neg']
        i+=1
    #print(train_X['header_sentiment_score'])
    
    k=0
    for x in text:
        sentiment_dict=sentiment.polarity_scores(x)
        train_X.at[k,'text_positive_score']=sentiment_dict['pos']
        train_X.at[k,'text_neutral_score']=sentiment_dict['neu']
        train_X.at[k,'text_negative_score']=sentiment_dict['neg']
        k+=1
    #print(train_X['text_sentiment_score'])

    
    #################################################
    
    #try textblob too######################
    '''
    j=0
    pol=[]
    sub=[]
    for x in text:
        temp_pol=TextBlob(x).sentiment.polarity
       # temp_sub=TextBlob(x).sentiment.subjectivity
        if temp_pol >=0.2:
            pol.append(1)
        elif temp_pol <=-0.2:
            pol.append(-1)
        else:
            pol.append(0)
        #sub.append(temp_sub)
    train_X['textblob_polarity']=pol
    #train_X['textblob_subjectivity']=sub
        
    '''


    #lengths of texts###############################
    
    train_X['review_length']=0
    
    m=0
    for x in df['review']:
        #print(len(x))
        train_X.at[m,'review_length']=len(x)
        m+=1
    
    #print(train_X['review_length'])
    
    #print("testing feature df!!!!!!!!!!!!!!!!")
    
    #print(train_X['text_positive_score'])
    #print(train_X['text_neutral_score'])
    #print(train_X['text_negative_score'])
    #print(train_X)
    
  #  train_X.fillna(0)
    return train_X
    ################################################




#join doc features and other features
def create_feature_df(df):#,vectorizer):
    X=create_doc_features(df)#,vectorizer)
    Y=create_other_feature_df(df)
    #Z=hstack([X,Y.to_numpy()])
    Z=pd.concat([X,Y],axis=1)
    print("combined df is ::::::::::")
    print(Z)
    return(Z)




#model building##################################


#prediction target
train_Y = train_df['labels']
valid_Y = valid_df['labels']

'''
print("tests!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print(train_df)
print(train_df['review'].head())
sent1=train_df['review'][4]
print(sent1)
'''
#feature dataframes#can be tweaked
train_X=create_feature_df(train_df)#,vectorizer)
valid_X=create_feature_df(valid_df)#,vectorizer)
test_X=create_feature_df(test_df)#,vectorizer)


#some tests
print(train_df)


#naive bayes from nltk to test feature importance
#test_classifier = nltk.NaiveBayesClassifier.train(train_X)
#test_classifier.show_most_informative_features(100)






# model testings and building

#decision tree
#model = tree.DecisionTreeClassifier(criterion='entropy',max_depth=(5),min_samples_split=1000)
#model = model.fit(train_X, train_Y)

#logistic regression
#model = LogisticRegression(max_iter=150,C=1.0)
#model.fit(train_X, train_Y)

#k-nearest neighbors
#model=KNeighborsClassifier()
#model.fit(train_X, train_Y)

#random forest
model=RandomForestClassifier(criterion='entropy',max_depth=50,min_samples_split=5,min_samples_leaf=5, n_estimators=50)
model.fit(train_X, train_Y)


'''
#feature importance list
important_features_dict = {}
for idx, val in enumerate(model.feature_importances_):
    important_features_dict[idx] = val

important_features_list = sorted(important_features_dict,
                                 key=important_features_dict.get,
                                 reverse=True)

#print('5 most important features:!!!!!!!!!!!!!!!!!!!!!!!!!!!!' )
#print(important_features_list[:1500])

#write features to text file, comment out if not writing new ones

improved_features=[]
for x in important_features_list[:4000]:
    if x <4000:
        improved_features.append(most_common_list[x])

textfile = open("improved_features.txt", "w")
for element in improved_features:
    textfile.write(element + ",")
textfile.close()

'''  






valid_Y_hat = model.predict(valid_X)
print(valid_Y)
print(valid_Y_hat)

#print results
train_Y_hat=model.predict(train_X)
print("on training set")
i=0
correct=0
for x in train_Y:
    if train_Y_hat[i] == x:
        correct+=1
    i+=1
result0=correct/len(train_Y)
print(result0)

print("on valid set")
i=0
correct=0
for x in valid_Y:
    if valid_Y_hat[i] == x:
        correct+=1
    i+=1
result1=correct/len(valid_Y)
print(result1)
    





