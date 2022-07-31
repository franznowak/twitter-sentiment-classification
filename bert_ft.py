import logging
import os

import datasets
import torch
from prettytable import PrettyTable

from bert import *
from transformers import AutoTokenizer

from util import set_trainable

logging.basicConfig(level=logging.INFO)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

FULL = False
FORCE = True

MODEL = 'distilbert-base-uncased'
TOKENIZER = 'bert-base-uncased'
FT = 'CLASS_LNORM'  # mode of fine-tuning
# 'CLASS_LNORM': train classifier and layer norm parameters
# 'PREC_CLASS': pre-classifier and classifier

EPOCHS = 2
BATCH_SIZE = 16

IDENT = '_'.join([MODEL, "ep", str(EPOCHS), "bs", str(BATCH_SIZE), str(FT), "2"])
DIR = "bert_data/" + IDENT

try:
    os.makedirs(DIR)
except FileExistsError:
    pass

dataset_train, dataset_val = load(full=FULL, preprocessing=None)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
train_tokenized = tokenize(
    dataset_train,
    tokenizer,
    path=f'bert/cache/train_tokenized__{TOKENIZER}{"__full" if FULL else ""}',
    force=FORCE)
val_tokenized = tokenize(
    dataset_val,
    tokenizer,
    path=f'bert/cache/val_tokenized__{TOKENIZER}{"__full" if FULL else ""}',
    force=FORCE)

model = get_BERT(MODEL, device)

for param in model.parameters():
    param.requires_grad = False

for param in model.classifier.parameters():
    param.requires_grad = True


set_trainable(model)

training_args = TrainingArguments(output_dir=DIR,
                                  num_train_epochs=EPOCHS,
                                  save_strategy="epoch",
                                  evaluation_strategy="epoch",
                                  per_device_train_batch_size=BATCH_SIZE,
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

val_pred = trainer.predict(val_tokenized)
y_pred = np.argmax(val_pred.predictions, axis=1)
y = val_tokenized.to_pandas()['label']
metrics = evaluate(y, y_pred)
