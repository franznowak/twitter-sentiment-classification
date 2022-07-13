from typing import Dict, Optional
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WhitespaceTokenizer
from nltk.corpus import stopwords

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


def preprocess(df: pd.DataFrame, flags: Optional[Dict[str, bool]]):
  if flags is not None:
    if flags['remove_tags']:
      remove_tags(df)
    if flags['tokenize']:
      tokenize(df)
    if flags['remove_stopwords']:
      remove_stopwords(df)
    if flags['lemmatize']:
      lemmatize(df)
