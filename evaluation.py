from cmath import log
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, roc_auc_score
from typing import Callable, Tuple
import logging


def _log_metrics(metrics):
  logging.info(
    '---\n' +
    '\n'.join([f'* {x}: {y}' for x,y in metrics.items()]) +
    '\n---'
  )


def evaluate_prob(y: np.array, y_pred: np.array, verbose=True) -> Tuple[float, float]:
  """
  Returns BCE loss, AUC in this order.
  """

  bce = log_loss(y, y_pred)
  auc = roc_auc_score(y, y_pred)
  if verbose:
    _log_metrics({'bce': bce, 'auc': auc})

  return bce, auc


def evaluate(y: np.array, y_pred: np.array) -> Tuple[float, float, float, float, float, float]:
  """
  Returns accuracy, precision, recall, F1, BCE loss, AUC in this order.

  * accuracy: proportion of correctly classified answers
  * precision: proportion of correctly classified positives
  * recall: proportion of actual positives correctly classified
  * F1: combination of precision & recall
  """

  accuracy = accuracy_score(y, y_pred)
  precision = precision_score(y, y_pred)
  recall = recall_score(y, y_pred)
  f1 = f1_score(y, y_pred)
  bce, auc = evaluate_prob(y, y_pred, verbose=False)
  _log_metrics({'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'bce': bce, 'auc': auc})

  return accuracy, precision, recall, f1, bce, auc


def evaluate_model(model: Callable[[pd.DataFrame], np.array], df: pd.DataFrame) -> Tuple[float, float, float, float, float, float]:
  """
  Expects a dataframe with columns `x` and `y`.
  """

  y = df['y'].to_numpy()
  y_pred = model(df)

  return evaluate(y, y_pred)

