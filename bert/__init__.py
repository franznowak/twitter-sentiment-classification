from pathlib import Path
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset, ClassLabel
from scipy.special import softmax

from evaluation import evaluate, evaluate_prob
from loading import load_train
from preprocessing import preprocess


def load(df_train, df_val, preprocessing=None):
  # df_train, df_val = load_train(full=full, eval_frac=0.2, x_col='text', y_col='label', neg_label=0, pos_label=1)

  preprocess(df_train, flags=preprocessing, x_col='text')
  preprocess(df_val, flags=preprocessing, x_col='text')

  dataset_train = Dataset.from_pandas(df_train)
  dataset_val = Dataset.from_pandas(df_val)

  new_features = dataset_train.features.copy()
  new_features['label'] = ClassLabel(names=['0', '1'])

  dataset_train = dataset_train.cast(new_features)
  dataset_val = dataset_val.cast(new_features)

  return dataset_train, dataset_val


def tokenize(ds, tokenizer, path=None, force=True):
  def tokenize_function(ds):
    return tokenizer(ds['text'], padding=True, truncation=True)

  def load_or_tokenize(ds, path, force):
    if not force and path is not None and Path(path).exists():
      return Dataset.load_from_disk(path)
    else:
      ds_tokenized = ds.map(tokenize_function, batched=True)
      if path is not None:
        ds_tokenized.save_to_disk(path)
      return ds_tokenized

  return load_or_tokenize(ds, path=path, force=force)


def get_BERT(model_name, device):
  model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, ignore_mismatched_sizes=True).to(device)
  model.save_pretrained(model_name)
  return model


def compute_metrics(eval_pred):
  predictions, labels = eval_pred

  point_estimates = np.argmax(predictions, axis=1)
  point_estimate_eval = evaluate(labels, point_estimates)

  prob_estimates = softmax(predictions, axis=1)[:, 1]
  prob_estimates_eval = evaluate_prob(labels, prob_estimates)
  confidence = np.max(prob_estimates, axis=1)
  all_confidence = confidence.mean()
  all_confidence_std = confidence.std()
  correct_confidence = confidence[labels == point_estimates].mean()
  correct_confidence_std = confidence[labels == point_estimates].std()
  incorrect_confidence = confidence[labels != point_estimates].mean()
  incorrect_confidence_std = confidence[labels != point_estimates].std()

  return {
    **point_estimate_eval,
    **prob_estimates_eval,
    'confidence': all_confidence,
    'confidence_std': all_confidence_std,
    'correct_confidence': correct_confidence,
    'correct_confidence_std': correct_confidence_std,
    'incorrect_confidence': incorrect_confidence,
    'incorrect_confidence_std': incorrect_confidence_std,
  }


def train(model_name, tokenizer_name, device, df_train, df_val, preprocessing=None, batch_size=32, epochs=1, force_tokenize=True):
  dataset_train, dataset_val = load(df_train, df_val, preprocessing=preprocessing)

  tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
  train_tokenized = tokenize(
    dataset_train,
    tokenizer,
    # path=f'bert/cache/train_tokenized__{tokenizer_name}{"__full" if full else ""}',
    force=force_tokenize)
  val_tokenized = tokenize(
    dataset_val,
    tokenizer,
    # path=f'bert/cache/val_tokenized__{tokenizer_name}{"__full" if full else ""}',
    force=force_tokenize)

  model = get_BERT(model_name, device)

  training_args = TrainingArguments(
    output_dir='bert_data/test_trainer',
    num_train_epochs=epochs,
    save_strategy='epoch',
    evaluation_strategy='epoch',
    per_device_train_batch_size=batch_size,
    load_best_model_at_end=True)

  trainer = Trainer(
    model,
    training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics)

  trainer.train()

  # val_pred = trainer.predict(val_tokenized)
  # y_pred = np.argmax(val_pred.predictions, axis=1)
  # y = val_tokenized.to_pandas()['label']
  # metrics = evaluate(y, y_pred)
  return model


def objective(args, model_name, tokenizer_name, device, full=False):
  print(args)
  _, metrics = train(model_name, tokenizer_name, device, full=full, **args)
  return -metrics['accuracy']
