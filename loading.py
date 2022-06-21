import pandas as pd
import os
from typing import List, Tuple

DIR = 'twitter-datasets'


def _read_data(path: str) -> List[str]:
  with open(path, 'r') as f:
    return [x for x in f]


def _read_data_with_ids(path: str) -> Tuple[List[str], List[str]]:
  index = []
  rows = []
  with open(path, 'r') as f:
    for line in f:
      id, x = line.split(',', maxsplit=1)
      index.append(id)
      rows.append(x)
  return index, rows


def load_train(full=False) -> pd.DataFrame:
  pos_path = os.path.join(DIR, 'train_pos' + ('_full' if full else '') + '.txt')
  neg_path = os.path.join(DIR, 'train_neg' + ('_full' if full else '') + '.txt')

  pos_rows = _read_data(pos_path)
  pos = pd.DataFrame({'x': pos_rows})
  pos['y'] = 1

  neg_rows = _read_data(neg_path)
  neg = pd.DataFrame({'x': neg_rows})
  neg['y'] = -1

  return pd.concat([pos, neg], ignore_index=True).reset_index()


def load_test() -> pd.DataFrame:
  path = os.path.join(DIR, 'test_data.txt')
  index, rows = _read_data_with_ids(path)
  df = pd.DataFrame({'x': rows}, index)

  return df
