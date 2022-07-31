import scipy
import scipy.special
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable


def count_parameters(model):
    table = PrettyTable(["Modules", "Trainable Parameters", "Parameters"])
    total_params_train = 0
    total_params = 0
    for name, parameter in model.named_parameters():
        params = parameter.numel()
        params_t = params
        if not parameter.requires_grad:
            params_t = 0
        table.add_row([name, params_t, params])
        total_params_train += params_t
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params_train}")
    pourcent = float(total_params_train)/total_params
    print(f"Percentage of trainable parameters: {pourcent}")

    return total_params_train, pourcent


def set_trainable(model):
    count = 0
    for name, parameter in model.named_parameters():
        if "layer_norm" in str(name) or "LayerNorm" in str(name):
            count += 1
            parameter.requires_grad = True

    print(count)
    return count_parameters(model)[0]


def analyze_misclass(y_pred_logits, y_pred, y_true, name1="boxplot", name2="histogram"):
    normed_logits = scipy.special.softmax(y_pred_logits, axis=1)
    df = pd.DataFrame({'log_1': normed_logits[:, 0], 'log_2': normed_logits[:, 1], 'y_pred': y_pred, 'y_true': y_true})
    df['log_diff'] = (df['log_1'] - df['log_2']).abs()
    df['diff_incorrect'] = df.loc[df['y_pred'] != df['y_true'], ['log_diff']]
    df['diff_correct'] = df.loc[df['y_pred'] == df['y_true'], ['log_diff']]
    mean_diff_correct = df['diff_correct'].mean()
    mean_diff_incorrect = df['diff_incorrect'].mean()

    plt.ioff()

    df.boxplot(column=['diff_correct', 'diff_incorrect'])
    plt.savefig(name1+".png")
    plt.close()
    plt.pause(0.1)

    plt.hist([df['diff_incorrect'], df['diff_correct']], bins=60, label=["Misclassified", "Correct"])
    plt.legend()
    plt.savefig(name2+".png")
    plt.close()

    return mean_diff_correct, mean_diff_incorrect, df

