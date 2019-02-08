import string
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
from collections import Counter
from nltk.corpus import wordnet

nltk.download('stopwords')
nltk.download('wordnet')

def text_process(text):
    # text = "string. With. Punctuation?"

    punc = str.maketrans({key: None for key in string.punctuation})
    text = text.translate(punc)
    word_list = text.split()

    filtered_words = [word for word in word_list if word.lower() not in stopwords.words('english')]
    filtered_words = [validWord for validWord in filtered_words if  wordnet.synsets(validWord)]
    return ' '.join(filtered_words)


def tokenizer(x):
    return x.split(',')
