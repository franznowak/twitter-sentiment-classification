import pandas as pd
import numpy as np
from typing import Callable

from loading import load_test


def prepare_model_submission(model: Callable[[pd.DataFrame], np.array], file = 'submission.csv'):
  df = load_test()
  y_pred = model(df)
  prepare_submission(y_pred, file)


def prepare_submission(y_pred: np.ndarray, file = 'submission.csv'):
  df = pd.DataFrame(y_pred, columns=['Prediction'])
  df.index += 1
  df.to_csv(file, index_label='Id')