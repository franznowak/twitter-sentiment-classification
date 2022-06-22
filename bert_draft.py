import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset, ClassLabel, load_metric
import pandas as pd

from loading import load_train, load_test

VERBOSE = False
TOKENIZER = "bert-base-uncased"  # todo
MODEL = "distilbert-base-uncased"  # todo
# name columns based on used model!! -> don't think that I have to do that/already correct

# LOAD DATA
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)  #, use_fast=True


def tokenize_function(ds):
    return tokenizer(ds['text'], padding=True, truncation=True)  # , return_tensors="pt"


df_train, df_val = load_train(full=False, dir="twitter-datasets", eval_frac=0.2, cols=["text", "label"])
dataset_train = Dataset.from_pandas(df_train)

if VERBOSE:
    print(dataset_train)
    print(dataset_train[0])
    print(dataset_train.features)

# todo: find way of defining the features correctly upon initialization
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
    return model


model = get_BERT()
training_args = TrainingArguments(output_dir="bert_data/test_trainer",
                                  num_train_epochs=1,
                                  save_strategy="epoch",
                                  evaluation_strategy="epoch",
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
trainer.evaluate()
