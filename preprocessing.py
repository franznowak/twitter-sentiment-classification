from typing import Dict, Optional
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WhitespaceTokenizer
from nltk.corpus import stopwords
from textblob import Word
from textblob import TextBlob
from tqdm import tqdm

nltk.download('stopwords')
nltk.download('omw-1.4')
stop_words = set(stopwords.words('english'))

nltk.download('wordnet')
tokenizer = WhitespaceTokenizer()
lemmatizer = WordNetLemmatizer()

tqdm.pandas()

def to_lower(df: pd.DataFrame, x_col='x'):
  """
  To be applied to a dataframe with a column called 'x' that contains sentences.
  """
  df[x_col] = df[x_col].apply(lambda sentence: sentence.lower())

def tokenize(df: pd.DataFrame, x_col='x'):
  """
  To be applied to a dataframe with a column called 'x' that contains sentences.
  """
  df[x_col] = df[x_col].apply(lambda sentence: tokenizer.tokenize(sentence))

def remove_tags(df: pd.DataFrame, x_col='x'):
  """
  To be applied to a dataframe with a column called 'x' that contains sentences.
  Deprecated in favour of remove_tag_tokens(df: pd.DataFrame)
  """
  df[x_col] = df[x_col].apply(lambda sentence: sentence.replace('<user>', '').replace('<url>', '').strip())

def remove_tag_tokens(df: pd.DataFrame, x_col='x'):
  """
  To be applied to a dataframe with a column called 'x' that contains tokens.
  """
  df[x_col] = df[x_col].apply(lambda tokens: [w for w in tokens if not w in ['user', '<url>']])

def remove_stopwords(df: pd.DataFrame, x_col='x'):
  """
  To be applied to a dataframe with a column called 'x' that contains tokens.
  """
  df[x_col] = df[x_col].apply(lambda tokens: [w for w in tokens if not w in stop_words])

def lemmatize(df: pd.DataFrame, x_col='x'):
  """
  To be applied to a dataframe with a column called 'x' that contains tokens.
  """
  df[x_col] = df[x_col].apply(lambda tokens: [lemmatizer.lemmatize(w) for w in tokens])

def remove_single_symbols(df: pd.DataFrame, x_col='x'):
  """
  To be applied to a dataframe with a column called 'x' that contains tokens.
  """
  df[x_col] = df[x_col].apply(lambda tokens: [w for w in tokens if len(w) > 1])

def spelling_correction(df: pd.DataFrame, x_col='x'):
  """
  To be applied to a dataframe with a column called 'x' that contains tokens.
  """
  df[x_col] = df[x_col].progress_apply(lambda tokens: [Word(w).correct() for w in tokens])


def replace_user_handles(df: pd.DataFrame, x_col='x'):
  """
  To be applied to a dataframe with a column called 'x' that contains tokens.
  """
  df[x_col] = df[x_col].apply(lambda tokens: [w if not (w.startswith("@") and len(w) > 1) else "<user>" for w in tokens])

def replace_urls(df: pd.DataFrame, x_col='x'):
  """
  To be applied to a dataframe with a column called 'x' that contains tokens.
  """
  df[x_col] = df[x_col].apply(lambda tokens: [w if not (w.startswith("http://") or w.startswith("https://") or w.startswith("www.")) else "<url>" for w in tokens])

def untokenize(df: pd.DataFrame, x_col='x'):
  """
  To be applied to a dataframe with a column called 'x' that contains tokens.
  """
  df[x_col] = df[x_col].apply(lambda tokens: " ".join(tokens))

def preprocess(df: pd.DataFrame, flags: Optional[Dict[str, bool]], x_col='x'):
  if flags is not None:
    if flags.get('to_lower', False):
      to_lower(df, x_col=x_col)
    if flags.get('tokenize', False):
      tokenize(df, x_col=x_col)
    if flags.get('replace_user_handles', False):
      replace_user_handles(df, x_col=x_col)
    if flags.get('replace_urls', False):
      replace_urls(df, x_col=x_col)  
    if flags.get('remove_tags', False):
      remove_tags(df, x_col=x_col)
    if flags.get('remove_tag_tokens', False):
      remove_tag_tokens(df, x_col=x_col)
    if flags.get('remove_stopwords', False):
      remove_stopwords(df, x_col=x_col)
    if flags.get('lemmatize', False):
      lemmatize(df, x_col=x_col)
    if flags.get('remove_single_symbols', False):
      remove_single_symbols(df, x_col=x_col)
    if flags.get('spelling_correction', False):
      spelling_correction(df, x_col=x_col)
