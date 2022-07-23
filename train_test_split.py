import pandas as pd

df_train = pd.read_csv('train_test/train.csv')
df_eval = pd.read_csv('train_test/test.csv')


def select_train(size=160_000):
  return df_train.iloc[:size].drop(['Unnamed: 0'], axis='columns')


def select_train_with_cluster(df_cluster_map: pd.DataFrame, cluster: int, size=160_000):
  df = pd.merge(df_train, df_cluster_map, on='index')
  return df[df['cluster'] == cluster].iloc[:size].drop(['Unnamed: 0'], axis='columns')


def select_eval(size=40_000):
  return df_eval.iloc[:size].drop(['Unnamed: 0'], axis='columns')


def select_eval_with_cluster(df_cluster_map: pd.DataFrame, cluster: int, size=40_000):
  df = pd.merge(df_eval, df_cluster_map, on='index')
  return df[df['cluster'] == cluster].iloc[:size].drop(['Unnamed: 0'], axis='columns')
