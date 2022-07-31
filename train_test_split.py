import os
import pandas as pd

DF_TRAIN = pd.read_csv(os.path.join(os.path.dirname(__file__), 'train_test/train.csv'))[['index', 'text', 'label']]
DF_EVAL = pd.read_csv(os.path.join(os.path.dirname(__file__), 'train_test/test.csv'))[['index', 'text', 'label']]


def select_train(size=160_000):
  df = DF_TRAIN
  if size is not None:
    df = df.iloc[:size]
  return df


def select_train_with_cluster(df_cluster_map: pd.DataFrame, cluster: int, size=160_000):
  df = pd.merge(DF_TRAIN, df_cluster_map, on='index')
  df = df[df['cluster'] == cluster]
  if size is not None:
    df = df.iloc[:size]
  return df


def select_eval(size=40_000):
  df = DF_EVAL
  if size is not None:
    df = df.iloc[:size]
  return df


def select_eval_with_cluster(df_cluster_map: pd.DataFrame, cluster: int, size=40_000):
  df = pd.merge(DF_EVAL, df_cluster_map, on='index')
  df = df[df['cluster'] == cluster]
  if size is not None:
    df = df.iloc[:size]
  return df
