import pandas as pd
import tensorflow as tf
import numpy as np
import dataPreProcessing as dpp
import string
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

# Import data from CSV
data = pd.read_csv('Reviews.csv')

data = data[['Text', 'Score']]

# Data Cleaning
for index, row in data.iterrows():
       text = str(row.Text)
       cleanString = dpp.text_process(text)
       data.loc[index , 'cleanedText']  = cleanString


data.to_csv("cleanData.csv", encoding='utf-8', index=False)