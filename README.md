# A Comparison of Techniques for Sentiment Classification

Authors: Jonas Hübotter, Franz Nowak, Fiona Muntwyler, Saahiti Prayaga

Date: 31st July 2022

This README provides an overview of the code developed for the 2022 spring semester CIL project.

If you try to run any of our code, make sure that you have all dependencies listed in `requirements.txt` installed.

## BERT

### BERT Warm Start

The BERT Warm Start models can be trained with the notebook `bert.ipynb`. The submissions are generated using `bert_submit.ipynb`. The models can be evaluated against the validation set using `bert_evaluate.ipynb`.

### BERT Fine-tuning

Fine-tuning was done in two different settings: in the first setting (O), only the parameters of the output layer were set to trainable, whereas in the second setting (LN), additionally the parameters of the layer norms were set to trainable.
The notebook to train and evaluate this fine-tuned BERT is `bert-finetuning.ipynb`.
The default setting is (LN). If you wish to change it to (O), the parameter `train_LN` has to be set to False.

### BERT Sub-networks

To train separate BERT models for each (or some) of the clusters (both unsupervised and emotion), the `bert_train_per_cluster.ipynb` notebook can be used.
For models that are composed of separate BERT models for each cluster, the notebooks `bert_submit_with_subnetworks.ipynb` and `bert_evaluate_with_subnetworks.ipynb` can be used to generate the submission and to evaluate them on a validation set, respectively.

## Naive Bayes

To replicate the results of the Naive Bayes classifier, use the notebook ``naive_bayes.ipynb``.

## XGBoost

The experiments of the XGBoost experiments can be found in ``xgboost_with_glove.ipynb``. The experiments with XGBoost on cluster 5 using BERT and fastText embeddings can be found in `xgboost_with_BERT.ipynb` and `xgboost_with_fasttext.ipynb`, respectively.

## Logistic Regression

To replicate the results of our logistic regression baseline, which uses fastText embeddings, please refer to the notebook ``LogReg+fastText.ipynb``
