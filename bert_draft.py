import datetime
import os

import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset, ClassLabel, load_metric

from evaluation import evaluate
from loading import load_train, load_test

VERBOSE = False
TOKENIZER = "bert-base-uncased"
MODEL = "distilbert-base-uncased"
# todo: use cardiffnlp/twitter-roberta-base-sentiment
EPOCHS = 1
TRAIN_BATCH_SIZE = 32
IDENT = '_'.join([MODEL, "ep", str(EPOCHS)])

# LOAD DATA
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)  # , use_fast=True


def tokenize_function(ds):
    return tokenizer(ds['text'], padding=True, truncation=True)  # , return_tensors="pt"


df_train, df_val = load_train(full=False, dir="twitter-datasets", eval_frac=0.2, cols=["text", "label"])
dataset_train = Dataset.from_pandas(df_train)

if VERBOSE:
    print(dataset_train)
    print(dataset_train[0])
    print(dataset_train.features)

new_features = dataset_train.features.copy()
new_features["label"] = ClassLabel(names=['0', '1'])
dataset_train = dataset_train.cast(new_features)

dataset_val = Dataset.from_pandas(df_val).cast(new_features)

if VERBOSE:
    print(dataset_train.features)
    print(dataset_val.features)

train_tokenized = dataset_train.map(tokenize_function, batched=True)
val_tokenized = dataset_val.map(tokenize_function, batched=True)

df_test = load_test(dir="twitter-datasets", cols="text")
dataset_test = Dataset.from_pandas(df_test)

test_tokenized = dataset_test.map(tokenize_function, batched=True)


# DEFINE MODEL
def get_BERT(model_name=MODEL):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.save_pretrained(model_name)
    return model


model = get_BERT()
training_args = TrainingArguments(output_dir="bert_data/test_trainer",
                                  num_train_epochs=EPOCHS,
                                  save_strategy="epoch",
                                  evaluation_strategy="epoch",
                                  per_device_train_batch_size=TRAIN_BATCH_SIZE,
                                  load_best_model_at_end=True,
                                  metric_for_best_model="accuracy")
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
    compute_metrics=compute_metrics
)

trainer.train()
# trainer.evaluate()

try:
    os.makedirs("bert_data")
except FileExistsError:
    pass
train_pred = trainer.predict(train_tokenized)
df_train['Prediction'] = np.argmax(train_pred.predictions, axis=1)
df_train.to_csv("bert_data/bert_pred_train.csv")
df_train['log_neg'] = train_pred.predictions[:, 0]
df_train['log_pos'] = train_pred.predictions[:, 1]
# store logits, e.g. for ensemble learning, ..
df_train.to_csv("bert_data/bert_pred_train_logits.csv")
acc_train, prec_train, recall_train, f1_train, bce_train, auc_train = evaluate(df_train['Prediction'], df_train[
    "label"])

val_pred = trainer.predict(val_tokenized)
df_val['Prediction'] = np.argmax(val_pred.predictions, axis=1)
df_val.to_csv("bert_data/bert_pred_val.csv")
df_val['log_neg'] = val_pred.predictions[:, 0]
df_val['log_pos'] = val_pred.predictions[:, 1]
df_val.to_csv("bert_data/bert_pred_val_logits.csv")
acc_val, prec_val, recall_val, f1_val, bce_val, auc_val = evaluate(df_val['Prediction'], df_val["label"])

test_pred = trainer.predict(test_tokenized)
df_test['Prediction'] = np.argmax(test_pred.predictions, axis=1)
df_test.to_csv("berts_data/bert_pred_test.csv")
df_test['log_neg'] = test_pred.predictions[:, 0]
df_test['log_pos'] = test_pred.predictions[:, 1]
df_test.to_csv("bert_data/bert_pred_test_logits.csv")
