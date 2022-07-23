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
