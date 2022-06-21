import pandas as pd
import os
from typing import List, Tuple, Union

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


def load_train(full=False, dir=DIR, eval_frac=None, cols=None) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    if cols is None:
        cols = ['x', 'y']
    pos_path = os.path.join(dir, 'train_pos' + ('_full' if full else '') + '.txt')
    neg_path = os.path.join(dir, 'train_neg' + ('_full' if full else '') + '.txt')

    pos_rows = _read_data(pos_path)
    pos = pd.DataFrame({cols[0]: pos_rows})
    pos[cols[1]] = 1

    neg_rows = _read_data(neg_path)
    neg = pd.DataFrame({cols[0]: neg_rows})
    neg[cols[1]] = 0

    df = pd.concat([pos, neg], ignore_index=True).reset_index()
    if eval_frac is None:
        return df

    else:
        val = df.sample(frac=eval_frac)
        train = df.drop(val.index)
        return train, val


def load_test(dir=DIR, cols='x') -> pd.DataFrame:
    path = os.path.join(dir, 'test_data.txt')
    index, rows = _read_data_with_ids(path)
    df = pd.DataFrame({cols: rows}, index)

    return df
