from pathlib import Path
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset, ClassLabel, load_metric

from evaluation import evaluate
from loading import load_train
from preprocessing import preprocess


def load(full=False, preprocessing=None):
  df_train, df_val = load_train(full=full, eval_frac=0.2, x_col='text', y_col='label', neg_label=0, pos_label=1)

  preprocess(df_train, flags=preprocessing)
  preprocess(df_val, flags=preprocessing)

  dataset_train = Dataset.from_pandas(df_train)
  dataset_val = Dataset.from_pandas(df_val)

  new_features = dataset_train.features.copy()
  new_features['label'] = ClassLabel(names=['0', '1'])

  dataset_train = dataset_train.cast(new_features)
  dataset_val = dataset_val.cast(new_features)

  return dataset_train, dataset_val


def tokenize(ds, tokenizer, path):
  def tokenize_function(ds):
    return tokenizer(ds['text'], padding=True, truncation=True)

  def load_or_tokenize(ds, path):
    if Path(path).exists():
      return Dataset.load_from_disk(path)
    else:
      ds_tokenized = ds.map(tokenize_function, batched=True)
      ds_tokenized.save_to_disk(path)
      return ds_tokenized

  return load_or_tokenize(ds, path=path)


def get_BERT(model_name, device):
  model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
  model.save_pretrained(model_name)
  return model


def train(model_name, tokenizer_name, device, full=False, preprocessing=None, batch_size=32, epochs=1):
  dataset_train, dataset_val = load(full=full, preprocessing=preprocessing)

  tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
  train_tokenized = tokenize(dataset_train, tokenizer, path=f'bert/cache/train_tokenized__{tokenizer_name}{"__full" if full else ""}')
  val_tokenized = tokenize(dataset_val, tokenizer, path=f'bert/cache/val_tokenized__{tokenizer_name}{"__full" if full else ""}')

  model = get_BERT(model_name, device)

  training_args = TrainingArguments(
    output_dir='bert_data/test_trainer',
    num_train_epochs=epochs,
    save_strategy='epoch',
    evaluation_strategy='epoch',
    per_device_train_batch_size=batch_size,
    load_best_model_at_end=True)

  metric = load_metric('accuracy')
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

  val_pred = trainer.predict(val_tokenized)
  y_pred = np.argmax(val_pred.predictions, axis=1)
  y = val_tokenized.to_pandas()['label']
  metrics = evaluate(y, y_pred)
  return model, metrics


def objective(args, model_name, tokenizer_name, device, full=False):
  print(args)
  _, metrics = train(model_name, tokenizer_name, device, full=full, **args)
  return -metrics['accuracy']
