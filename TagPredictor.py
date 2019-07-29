# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 10:46:10 2018

@author: sumanth.reddy
"""

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import sqlite3
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from wordcloud import WordCloud
import re
import os
from sqlalchemy import create_engine # database connection
import datetime as dt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.metrics import f1_score,precision_score,recall_score
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from skmultilearn.adapt import mlknn
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import LabelPowerset
from sklearn.naive_bayes import GaussianNB
from datetime import datetime

##Data Loading and Cleaning########

##Using Pandas with SQLite to Load the data
if not os.path.isfile('trainingData.db'):
    start = datetime.now()
    disk_engine= create_engine("sqlite:///trainingData.db")
    chunksize = 10000
    start_index = 1
    j=0
    for df in pd.read_csv('Train_lite.csv',iterator = True,names=['Id','Title','Body','Tags'],chunksize=chunksize,encoding='latin-1'):
        df.index += start_index
        j+=1
        print('Number of rows in database: {}'.format(j*chunksize))
        df.to_sql('data',disk_engine,if_exists='append')
        start_index = df.index[-1]+1
    print('Time taken to run this cell :',datetime.now()-start)
   
#Counting number of rows        
if os.path.isfile('trainingData.db'):
    start = datetime.now()
    con = sqlite3.connect('trainingData.db')
    rows = pd.read_sql_query('SELECT count(*) from data',con)
    print('Total number of rows: ',rows["count(*)"].values[0])
    con.close()

    
#Checking duplicates

if os.path.isfile('trainingData.db'):
    con = sqlite3.connect('trainingData.db')
    no_dup = pd.read_sql_query('SELECT Title,Body,Tags,count(*) as cnt_dup from data GROUP BY Title,Body,Tags',con)
    print('Number of non duplicate rows: ',no_dup.shape[0])
    con.close()

#Tag Count
count = no_dup.cnt_dup.value_counts()
print(count)

#Creating new database with 'without duplicates data'
if not os.path.isfile('trainingData_no_duplicates.db'):
    disk_engine = create_engine('sqlite:///trainingData_no_duplicates.db')
    no_dup_df = pd.DataFrame(no_dup,columns=['Title','Body','Tags'])
    no_dup_df.to_sql('data_no_dup',disk_engine,if_exists='append')

#TagData extraction
if os.path.isfile('trainingData_no_duplicates.db'):
    con = sqlite3.connect('trainingData_no_duplicates.db')
    tagData = pd.read_sql_query('SELECT Tags from data_no_dup',con)
    tagData.drop(tagData.index[0])
    print(tagData.head())
    con.close()
      
 #Analysis of Tags
vectorizer = CountVectorizer(tokenizer= lambda x : x.split())
tqdm = vectorizer.fit_transform(tagData['Tags'])
print('Total rows in db: ',tqdm.shape[0])
print('Total unique tags: ',tqdm.shape[1])

#top 10 tags 
tags = vectorizer.get_feature_names()
print(tags[:10])   

#frequenccy of each tag
freq = tqdm.sum(axis=0).A1
result = dict(zip(tags,freq))
print(result)

#saving dictionary to dataframe
if not os.path.isfile('tag_count.csv'):
    with open('tag_count.csv','w') as csvfile:
        writer = csv.writer(csvfile)
        for key,value in result.items():
            writer.writerow([key,value])
tag_count_df = pd.read_csv('tag_count.csv',names=['Tags','Count'])
print(tag_count_df.head())

#sort tag values
tg_sorted = tag_count_df.sort_values(['Count'],ascending = False)
values = tg_sorted['Count'].values

plt.plot(values)
plt.grid()
plt.title('Distribution of tags')
plt.xlabel('Tag number')
plt.ylabel('Number of times tag appeared')
plt.show()

plt.plot(values[:10000])
plt.grid()
plt.title('Distribution of first 10k tags')
plt.xlabel('Tag number')
plt.ylabel('Number of times tag appeared')
plt.show()

plt.plot(values[:5000])
plt.grid()
plt.title('Distribution of first 5k tags')
plt.xlabel('Tag number')
plt.ylabel('Number of times tag appeared')
plt.show()
    
plt.plot(values[:500])
plt.grid()
plt.title('Distribution of first 500 tags')
plt.xlabel('Tag number')
plt.ylabel('Number of times tag appeared')
plt.show()
    
    
lst_tags_10k = tag_count_df[tag_count_df.Count > 100].Tags
print('Tags used more than 100 times are: {}'.format(len(lst_tags_10k)))    

tag_question_count = tqdm.sum(axis=1).tolist()
print(tag_question_count[:10])

#converting list of lists to list
tag_question_count = [int(j) for i in tag_question_count for j in i]
print(tag_question_count[:10])
   
sns.countplot(tag_question_count,palette = 'gist_rainbow')
plt.title('Tags per question')
plt.xlabel('Tags')
plt.ylabel('Number of questions')
plt.show()    

#wordcloud
tup = dict(result.items())
print(tup)
wordcloud = WordCloud(background_color='yellow',width=300,height=200).generate_from_frequencies(tup)
fig = plt.figure(figsize=(30,20))
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout(pad=0)
fig.savefig("tag.png")
plt.show()   
    
i = np.arange(30)
tg_sorted.head(30).plot(kind='bar')
plt.title('top 30 tags')
plt.xticks(i,tg_sorted['Tags'])
plt.xlabel('Tags')
plt.ylabel('Count')
plt.show()

#Preprocessing of questions 

#db connections

def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)
 
    return None

def create_table(conn, create_table_sql):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)
        
def checkTableExists(dbcon):
    cursr = dbcon.cursor()
    str = "select name from sqlite_master where type='table'"
    table_names = cursr.execute(str)
    print("Tables in the databse:")
    tables =table_names.fetchall() 
    print(tables[0][0])
    return(len(tables))

def create_database_table(database, query):
    conn = create_connection(database)
    if conn is not None:
        create_table(conn, query)
        checkTableExists(conn)
    else:
        print("Error! cannot create the database connection.")
    conn.close()

sql_create_table = """CREATE TABLE IF NOT EXISTS QuestionsProcessed (question text NOT NULL, code text, tags text, words_pre integer, words_post integer, is_code integer);"""
create_database_table("Processed.db", sql_create_table)



start = datetime.now()
read_db = 'trainingData_no_duplicates.db'
write_db = 'Processed.db'
if os.path.isfile(read_db):
    conn_r = create_connection(read_db)
    if conn_r is not None:
        reader =conn_r.cursor()
        reader.execute("SELECT Title, Body, Tags From data_no_dup ORDER BY RANDOM() LIMIT 100000;")

if os.path.isfile(write_db):
    conn_w = create_connection(write_db)
    if conn_w is not None:
        tables = checkTableExists(conn_w)
        writer =conn_w.cursor()
        if tables != 0:
            writer.execute("DELETE FROM QuestionsProcessed WHERE 1")
            print("Cleared All the rows")
print("Time taken to run this cell :", datetime.now() - start)

#html code remover
from nltk.corpus import stopwords
def striphtml(data):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr,' ',str(data))
    return cleantext

stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')

#processing question 

start = datetime.now()
preprocessed_data_list=[]
reader.fetchone()
questions_with_code=0
len_pre=0
len_post=0
questions_proccesed = 0
for row in reader:

    is_code = 0

    title, question, tags = row[0], row[1], row[2]

    if '<code>' in question:
        questions_with_code+=1
        is_code = 1
    x = len(question)+len(title)
    len_pre+=x

    code = str(re.findall(r'<code>(.*?)</code>', question, flags=re.DOTALL))

    question=re.sub('<code>(.*?)</code>', '', question, flags=re.MULTILINE|re.DOTALL)
    question=striphtml(question.encode('utf-8'))

    title=title.encode('utf-8')

    question=str(title)+" "+str(question)
    question=re.sub(r'[^A-Za-z]+',' ',question)
    words=word_tokenize(str(question.lower()))

    #Removing all single letter and and stopwords from question exceptt for the letter 'c'
    question=' '.join(str(stemmer.stem(j)) for j in words if j not in stop_words and (len(j)!=1 or j=='c'))

    len_post+=len(question)
    tup = (question,code,tags,x,len(question),is_code)
    questions_proccesed += 1
    writer.execute("insert into QuestionsProcessed(question,code,tags,words_pre,words_post,is_code) values (?,?,?,?,?,?)",tup)
    if (questions_proccesed%100000==0):
        print("number of questions completed=",questions_proccesed)

no_dup_avg_len_pre=(len_pre*1.0)/questions_proccesed
no_dup_avg_len_post=(len_post*1.0)/questions_proccesed

print( "Avg. length of questions(Title+Body) before processing: %d"%no_dup_avg_len_pre)
print( "Avg. length of questions(Title+Body) after processing: %d"%no_dup_avg_len_post)
print ("Percent of questions containing code: %d"%((questions_with_code*100.0)/questions_proccesed))

print("Time taken to run this cell :", datetime.now() - start)


conn_r.commit()
conn_w.commit()
conn_r.close()
conn_w.close()


if os.path.isfile(write_db):
    conn_r = create_connection(write_db)
    if conn_r is not None:
        reader =conn_r.cursor()
        reader.execute("SELECT question From QuestionsProcessed LIMIT 10")
        print("Questions after preprocessed")
        print('='*100)
        reader.fetchone()
        for row in reader:
            print(row)
            print('-'*100)
conn_r.commit()
conn_r.close()

#Taking 1 Million entries to a dataframe.
write_db = 'Processed.db'
if os.path.isfile(write_db):
    conn_r = create_connection(write_db)
    if conn_r is not None:
        preprocessed_data = pd.read_sql_query("""SELECT question, Tags FROM QuestionsProcessed""", conn_r)
conn_r.commit()
conn_r.close()

print("number of data points in sample :", preprocessed_data.shape[0])
print("number of dimensions :", preprocessed_data.shape[1])

#Machine Learning Models

#Converting tags for multilabel problems 
# binary='true' will give a binary vectorizer
vectorizer = CountVectorizer(tokenizer = lambda x: x.split(), binary='true')
multilabel_y = vectorizer.fit_transform(preprocessed_data['tags'])

#ex. top 100 tags ,top 500 tags 
def tags_to_choose(n):
    t = multilabel_y.sum(axis=0).tolist()[0] #[0] because list of lists hence taking inner list 
    sorted_tags_i = sorted(range(len(t)), key=lambda i: t[i], reverse=True)
    multilabel_yn=multilabel_y[:,sorted_tags_i[:n]] # : says all question labels and sorted_tags top n tags
    return multilabel_yn

#ex. for top x number of tags how many questions are covered 
def questions_explained_fn(n):
    multilabel_yn = tags_to_choose(n)
    x= multilabel_yn.sum(axis=1)
    return (np.count_nonzero(x==0))

#checking questions covered for top n tags
questions_explained = []
total_tags=multilabel_y.shape[1]
total_qs=preprocessed_data.shape[0]
for i in range(100, total_tags, 100):
    questions_explained.append(np.round(((total_qs-questions_explained_fn(i))/total_qs)*100,3))

fig, ax = plt.subplots()
ax.plot(questions_explained)
xlabel = list(100+np.array(range(-50,450,50))*50)
ax.set_xticklabels(xlabel)
plt.xlabel("Number of tags")
plt.ylabel("Number Questions coverd partially")
plt.grid()
plt.show()
# you can choose any number of tags based on your computing power, minimun is 50(it covers 90% of the tags)
print("with ",17600,"tags we are covering ",questions_explained[100],"% of questions")

multilabel_yx = tags_to_choose(19600)
print("number of questions that are not covered :", questions_explained_fn(19600),"out of ", total_qs)

print("Number of tags in sample :", multilabel_y.shape[1])
print("number of tags taken :", multilabel_yx.shape[1],"(",(multilabel_yx.shape[1]/multilabel_y.shape[1])*100,"%)")


#Split data into 80:20 train-test 
total_size=preprocessed_data.shape[0]
train_size=int(0.80*total_size)

x_train=preprocessed_data.head(train_size)
x_test=preprocessed_data.tail(total_size - train_size)

y_train = multilabel_yx[0:train_size,:]
y_test = multilabel_yx[train_size:total_size,:] 


print("Number of data points in train data :", y_train.shape)
print("Number of data points in test data :", y_test.shape)


#featurizing data(questions)
start = datetime.now()
vectorizer = TfidfVectorizer(min_df=0.00009, max_features=200000, smooth_idf=True, norm="l2", \
                             tokenizer = lambda x: x.split(), sublinear_tf=False, ngram_range=(1,3))
x_train_multilabel = vectorizer.fit_transform(x_train['question'])
x_test_multilabel = vectorizer.transform(x_test['question'])
print("Time taken to run this cell :", datetime.now() - start)

print("Dimensions of train data X:",x_train_multilabel.shape, "Y :",y_train.shape)
print("Dimensions of test data X:",x_test_multilabel.shape,"Y:",y_test.shape)


classifier = OneVsRestClassifier(SGDClassifier(loss='log', alpha=0.00001, penalty='l1'), n_jobs=-1)
classifier.fit(x_train_multilabel, y_train)
predictions = classifier.predict(x_test_multilabel)

print("accuracy :",metrics.accuracy_score(y_test,predictions))
print("macro f1 score :",metrics.f1_score(y_test, predictions, average = 'macro'))
print("micro f1 scoore :",metrics.f1_score(y_test, predictions, average = 'micro'))
print("hamming loss :",metrics.hamming_loss(y_test,predictions))
print("Precision recall report :\n",metrics.classification_report(y_test, predictions))







