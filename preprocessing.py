import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WhitespaceTokenizer
from nltk.corpus import stopwords
from textblob import Word

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

nltk.download('wordnet')
tokenizer = WhitespaceTokenizer()
lemmatizer = WordNetLemmatizer()

def remove_tags(df: pd.DataFrame):
  df['x'] = df['x'].apply(lambda x: x.replace('<user>', '').replace('<url>', '').strip())

def tokenize(df: pd.DataFrame):
  df['x'] = df.x.apply(lambda sentence: tokenizer.tokenize(sentence))

def remove_stopwords(df: pd.DataFrame):
  df['x'] = df.x.apply(lambda tokens: [w for w in tokens if not w in stop_words])

def lemmatize(df: pd.DataFrame):  
  df['x'] = df.x.apply(lambda tokens: [lemmatizer.lemmatize(w) for w in tokens])

def remove_single_symbols(df: pd.DataFrame):
  df['x'] = df.x.apply(lambda tokens: [w for w in tokens if len(w) > 1])

def spelling_correction(df: pd.DataFrame):
  df['x'] = df.x.apply(lambda tokens: [Word(w).correct() for w in tokens])
