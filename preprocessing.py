import pandas as pd


def remove_tags(df: pd.DataFrame):
  df['x'] = df['x'].apply(lambda x: x.replace('<user>', '').replace('<url>', '').strip())
