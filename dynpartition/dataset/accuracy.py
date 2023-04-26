import torch


def sentiment_accuracy_score(predictions, labels):
    correct = torch.isclose(predictions, labels).sum()
    total = labels.size(0)
    acc = float(correct) / total
    return acc
