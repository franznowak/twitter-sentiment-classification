from pathlib import Path
import logging
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset, ClassLabel, load_metric

from evaluation import evaluate
from loading import load_train, load_test


def load(full=False, preprocessing=None):
  df_train, df_val = load_train(full=full, eval_frac=0.2, x_col='text', y_col='label', neg_label=0, pos_label=1)
  dataset_train = Dataset.from_pandas(df_train)
  dataset_val = Dataset.from_pandas(df_val)

  new_features = dataset_train.features.copy()
  new_features['label'] = ClassLabel(names=['0', '1'])

  dataset_train = dataset_train.cast(new_features)
  dataset_val = dataset_val.cast(new_features)

  return dataset_train, dataset_val


def tokenize(ds, tokenizer, path, force_retokenize=False):
  def tokenize_function(ds):
    return tokenizer(ds['text'], padding=True, truncation=True)

  def load_or_tokenize(ds, path, force_retokenize=False):
    if not force_retokenize and Path(path).exists():
      return Dataset.load_from_disk(path)
    else:
      ds_tokenized = ds.map(tokenize_function, batched=True)
      ds_tokenized.save_to_disk(path)
      return ds_tokenized

  return load_or_tokenize(ds, path=path, force_retokenize=force_retokenize)


def get_BERT(model_name, device):
  model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
  model.save_pretrained(model_name)
  return model


def train(model_name, tokenizer_name, device, full=False, force_retokenize=False, preprocessing=None, batch_size=32, epochs=1):
  dataset_train, dataset_val = load(full=full, preprocessing=preprocessing)

  tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
  train_tokenized = tokenize(dataset_train, tokenizer, path='bert/train_tokenized', force_retokenize=force_retokenize)
  val_tokenized = tokenize(dataset_val, tokenizer, path='bert/val_tokenized', force_retokenize=force_retokenize)

  model = get_BERT(model_name, device)

  training_args = TrainingArguments(
    output_dir="bert_data/test_trainer",
    num_train_epochs=epochs,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    per_device_train_batch_size=batch_size,
    load_best_model_at_end=True)

  metric = load_metric("accuracy")
  def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

  trainer = Trainer(
    model,
    training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics)

  trainer.train()


def objective(args, model_name, tokenizer_name, device, full=False, force_retokenize=False):
  return train(model_name, tokenizer_name, device, full=full, force_retokenize=force_retokenize, **args)
