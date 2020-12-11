#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 13:07:22 2020

I'm going to build model that predicts Wine Enthusiast scores for a bunch of
wines. I'm using the following data set from Kaggle...

https://www.kaggle.com/zynicide/wine-reviews?select=winemag-data_first150k.csv

This data was initially scrapped from the Wine Enthusiast website. My plan 
is to incorporate NLP and text analytics, since the data set has 
descriptions for each wine.

@author: Matt
"""

"""
The first step in the process is to import all of the libraries that will 
be needed. Since I'm still in the data exploration phase, I'm only 
importing pandas and numpy for now.

"""

# Import libraries
import pandas as pd
import numpy as np
import nltk
import re


# Download text corpora
nltk.download()

# List out text corpora
from nltk.book import *
from nltk.corpus import treebank

"""
This data set consists of 2 csv files. One with 150k records and one with 
130k records. After initially exploring the data, I found that there were 
two things in the csv files that need to be fixed. 

In one case, there were quotations marks in the description field. This 
caused a problem with uploading the data into a spark dataframe, because 
it was splitting the description column when it shouldn't have. I tried 
fixing this using .open(), .writefiles(), and other python functions. 
Unfortunately, I wasn't able to figure out a way to remove the quotation 
marks, and save the file as a usable csv. Maybe that's something I can 
figure out later, but will skip for this captstone.

The other issue was that there was a record with '\r\n' in the description.
I don't know why, but I removed that as well.

The way I did this was to import the csv into a pandas dataframe, and apply
a replace function on the records. If I was dealing with a dataset large 
enough, this may not work, which is why it would be smart to figure this 
out using .open() etc. However, for now, I made the necessary adjustments 
and saved them as new csv files. These new files will be used for the spark 
dataframes.

"""

# Load 150k file into pandas dataframe
wine150k = pd.read_csv('/Users/Matt/Desktop/DataScience/TextPractice/WineReviews/winemag-data_first150k.csv')

# remove quotations marks for all description records
wine150k['description'] = wine150k['description'].apply(lambda row: row.replace('"', ''))
# one record has ''\r\n' in the description that needs to be removed
wine150k['description'] = wine150k['description'].apply(lambda row: row.replace('\r\n', ' '))


# Load 130k file into pandas dataframe
wine130k = pd.read_csv('/Users/Matt/Desktop/DataScience/TextPractice/WineReviews/winemag-data_130k_v2.csv')

# remove quotations marks for all description records
wine130k['description'] = wine130k['description'].apply(lambda row: row.replace('"', ''))


"""
Since the 130k dataframe has 3 additional fields, I'll need to remove those
before combining the data. Once I do that, I want to make sure I remove 
any duplicates between the two files.

I found a df.drop() function that does remove columns, but when I use 
df.show() right after, the changes don't seem to stick. However, if I 
combine the dataframes using unionAll, but use df.drop() in the statement,
my new wine_combined dataframe seems to be created without any issue. 
I'm not sure if that's the best way to code it, but I'll stay with that 
for now.

I also found the df.distinct() function to remove duplicate rows from the 
dataframe. Similar to df.drop(), I don't know if the changes stick. Worst 
case, I can probably just create a new dataframe from this.

I'm also renaming the 'Unnamed: 0' column to 'ID', just so it's a cleaner 
column name to work with.

"""

# Remove columns in 130k file that aren't in 150k file
# https://sparkbyexamples.com/spark/spark-drop-column-from-dataframe-dataset/
wine130k.drop(['taster_name','taster_twitter_handle','title'], axis=1, inplace=True)

# Combine both wine files into one dataframe
# https://stackoverflow.com/questions/40397206/how-can-i-combineconcatenate-two-data-frames-with-the-same-column-name-in-java
wine_combined = pd.concat([wine150k, wine130k])

wine_combined['Above_90'] = np.where(wine_combined.points > 90, 1, 0)

# Rename column 'Unnamed: 0' to 'ID'
# https://sparkbyexamples.com/spark/rename-a-column-on-spark-dataframes/#rename-column
# wine_combined = wine_combined.withColumnRenamed("Unnamed: 0","ID")

# Remove any duplicate rows in dataframe
# https://sparkbyexamples.com/spark/spark-remove-duplicate-rows/
# wine_combined.distinct().show()


# Frequency of words in personals corpus
dist = FreqDist(text8) # text8 is personals

# Unique words
vocab1 = dist.keys()

# Words appearing > 100 times
frequent_words = [v for v in vocab1 if dist[v] > 10 and len(v) > 5]

# Normalize and Stem words
porter = nltk.PorterStemmer()

# Lemmatization - Stemming but results need to be valid words
WNLem = nltk.WordNetLemmatizer()


# NLTK built in tokenizers
sent_fox = "The fox jumped over the brown lazy dog on his way to eat the chicken nuggets!"
WToken = nltk.word_tokenize(sent_fox)

text = wine_combined.iloc[10].description
SToken = nltk.sent_tokenize(text)

""" Part of Speech Tagging """

# Figure out what type of word 'MD' is 
nltk.help.upenn_tagset('MD')

# Get parts of speech for every word in WToken
nltk.pos_tag(WToken)

""" Parsing Sentence Structure """

# NEED TO RESEARCH HOW TO DO THIS FOR LONGER TEXT!!!
nltk.CFG.fromstring("""
                    S -> NP VP
                    VP -> V NP | VP PP
                    PP -> P NP
                    """)
                    
# Can do this with treebank
text_treebank = treebank.parsed_sents('wsj_0001.mrg')[0]
print(text_treebank)


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(wine_combined['description']
,wine_combined['Above_90']
,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer


# ngram_range will take features that are words adjacent to each other
# ie. 'really good', 'very bad', etc.
# can set this to > 2 as well!
wine_vect = CountVectorizer(min_df=10, ngram_range=(1,3)).fit(x_train)

# Get number of words (features) in all of the wine reviews
len(wine_vect.get_feature_names())

# Create sparse matrix with all the words in the wine reviews
# Each row will represent each wine review
# each column will be 0 or > 0 depending on if the word is in
# that specific review!
x_train_vectorized = wine_vect.transform(x_train)


from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=10000)
model.fit(x_train_vectorized, y_train)

from sklearn.metrics import roc_auc_score

predictions = model.predict(wine_vect.transform(x_test))
probabilities = model.predict_proba(wine_vect.transform(x_test))

print('Area Under Curve:', roc_auc_score(y_test, predictions))
# Area Under Curve: 0.8287233120672829

# get all feature names included in model
feature_names = np.array(wine_vect.get_feature_names())

sorted_coef_index = model.coef_[0].argsort()

print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))

print('Largest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:-11:-1]]))


# Import TFIDF if you want to weight more common features vs others
# For this model, we'll import it to set a min df
from sklearn.feature_extraction.text import TfidfVectorizer

# We'll require features to show up in at least 10 reviews
# This reduces features from 34,960 to 11,454
# Turns out AUC scores for min_df=3,5,10 are all lower
new_vect = TfidfVectorizer(min_df=5).fit(x_train)

#x_train_vectorized = new_vect.transform(x_train)


# https://gist.github.com/larsmans/3745866
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
vect_df = pd.DataFrame(x_train_vectorized.toarray(), columns=wine_vect.get_feature_names())


import re

df92 = wine_combined[wine_combined.description.str.contains("92")].copy()
df92['description'] = df92.description.str.replace(r'[0-9]+[-]?[‚Äì]?[0-9]+', '')
df92.to_csv(r'/Users/Matt/Desktop/DataScience/TextPractice/WineReviews/test.csv', index=False)
df92.description.str.findall(r'[0-9]+[-][0-9]+')


#Replace all white-space characters with the digit "9":

txt = "The rain in Spain"
x = re.sub("\s", "9", txt)
print(x)






















