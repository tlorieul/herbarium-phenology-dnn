from collections import OrderedDict

import numpy as np
import torch


def train(model, optimizer, criterion, train_data_loader, n_epochs,
          lr_scheduler=None, val_data_loader=None, metrics=[],
          verbose=True, gpu=False, callbacks=[]):

    if gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    history = []

    model.train()
    model.to(device)

    for callback in callbacks:
        callback.set_model(model)
        callback.on_train_start()

    for i in range(n_epochs):
        for callback in callbacks:
            callback.on_epoch_start(i)

        if lr_scheduler:
            lr_scheduler.step()

        train_metrics = OrderedDict(
            [('loss', 0)]
            + [(metric.__name__, 0) for metric in metrics]
        )

        n_samples = 0

        for inputs, targets in train_data_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            for callback in callbacks:
                callback.on_batch_start(inputs, targets)

            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()

            optimizer.step()

            train_metrics['loss'] += loss.data * inputs.size(0)

            for metric in metrics:
                res = metric(outputs, targets)
                train_metrics[metric.__name__] += res * inputs.size(0)

            n_samples += inputs.size(0)

            for callback in callbacks:
                callback.on_batch_end(inputs, targets, outputs, loss)

        history.append(OrderedDict())

        for m in train_metrics:
            history[-1][m] = (train_metrics[m] / n_samples).item()

        if val_data_loader:
            with torch.no_grad():
                model.eval()

                val_metrics = dict(
                    [('loss', 0)]
                    + [(metric.__name__, 0) for metric in metrics]
                )
                n_val_samples = 0
                for inputs, targets in val_data_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_metrics['loss'] += loss.data * inputs.size(0)
                    for metric in metrics:
                        res = metric(outputs, targets)
                        val_metrics[metric.__name__] += res * inputs.size(0)
                    n_val_samples += inputs.size(0)
                for m in train_metrics:
                    history[-1]['val_' + m] = \
                        (val_metrics[m] / n_val_samples).item()

                model.train()

        if verbose:
            print('epoch={:<5} {}'.format(
                i+1,
                ' '.join(['{}={:<10.5f}'.format(*h)
                          for h in history[-1].items()])
            ))

        for callback in callbacks:
            callback.on_epoch_end(i, history)

    for callback in callbacks:
        callback.on_train_end(history)

    model.eval()

    history = dict(zip(history[0], zip(*[d.values() for d in history])))
    return history


def predict(model, data_loader, final_activation=None, gpu=False,
            force_model_eval_mode=True):
    if gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model.to(device)

    if force_model_eval_mode:
        model.eval()

    with torch.no_grad():
        preds = []

        for inputs, _ in data_loader:
            inputs = inputs.to(device)

            predictions = model(inputs)
            if final_activation is not None:
                predictions = final_activation(predictions)

            preds.append(predictions.cpu().numpy())

        preds = np.vstack(preds)

    return preds


def _check_is_probability(x):
    if x.size(1) == 1:
        return (x >= 0).all() and (x <= 1).all()
    else:
        return ((x >= 0).all() and (x <= 1).all()
                and (x.sum(axis=-1) == 1.).all())


def binary_accuracy(outputs, targets):
    if _check_is_probability(outputs):
        preds = outputs >= .5
    else:
        preds = outputs >= 0
    return (preds == targets.byte()).float().mean()


def multiclass_accuracy(outputs, targets):
    if outputs.size(1) == 1:
        return binary_accuracy(outputs, targets)
    else:
        preds = torch.max(outputs, dim=1)[1]
        s = (preds == targets).float().sum(0)
        return s / outputs.size(0)
