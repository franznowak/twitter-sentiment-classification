import logging
import torch
from bert import *
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

FULL = False
FORCE = True

MODEL = 'distilbert-base-uncased'
TOKENIZER = 'bert-base-uncased'

EPOCHS = 1
BATCH_SIZE = 32

IDENT = '_'.join([MODEL, "ep", str(EPOCHS)])

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
print(model)

for param in model.parameters():
    param.requires_grad = False

for param in model.pre_classifier.parameters():
    param.requires_grad = True

for param in model.classifier.parameters():
    param.requires_grad = True

training_args = TrainingArguments(output_dir="bert_data/test_trainer",
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
