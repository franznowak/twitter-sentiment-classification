# A Comparison of Techniques for Sentiment Classification

Authors: Jonas Hübotter, Franz Nowak, Fiona Muntwyler, Saahiti Prayaga

Date: 31st July 2022

This Readme provides an overview of the code developped for the 2022 spring semester CIL project.

If you try to run any of our code, make sure that you have all dependencies listed in `requirements.txt` installed.

## BERT
### BERT warm start

### BERT fine-tuning
Fine-tuning was done in two different settings: in the first setting (O), only the parameters of the output layer were set to trainable, whereas in the second setting (LN), additionally the parameters of the layer norms were set to trainable.
The notebook to train and evaluate this fine-tuned BERT is `bert-finetuning.ipynb`.
The default setting is (LN). If you wish to change it to (O), the parameter `train_LN` has to be set to False.


## Naive Bayes

## XGBoost

## Logistic Regression