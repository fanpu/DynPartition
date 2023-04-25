from pathlib import Path

import lovely_tensors as lt
import torch
from torch import nn

from dynpartition.dataset.sst_dataset import SSTDataset
from dynpartition.models.TreeLSTM import TreeLSTMSentiment


def load_tree_lstm():
    num_classes = 3
    input_dim = 300
    mem_dim = 150
    data_folder = "_data"

    # vocab_file = "vocab-cased.pth"
    train_file = "sst_train_constituency_state_dict.pth"
    dev_file = "sst_dev_constituency_state_dict.pth"
    test_file = "sst_test_constituency_state_dict.pth"
    model_file = "20230424221345_constituency_model_12.pth"
    embedding_file = "20230424221345_constituency_embedding_12.pth"

    base_path = Path(__file__)
    while base_path.name != "dynpartition":
        base_path = base_path.parent

    base_path = base_path.joinpath(data_folder)

    print("Loading TreeLSTM model...")

    vocab_size = 21699
    # vocab_size = Vocab().load_state_dict(torch.load(base_path.joinpath(vocab_file))).size()
    train_dataset = SSTDataset().load_state_dict(torch.load(base_path.joinpath(train_file)))
    dev_dataset = SSTDataset().load_state_dict(torch.load(base_path.joinpath(dev_file)))
    test_dataset = SSTDataset().load_state_dict(torch.load(base_path.joinpath(test_file)))

    embedding_model = nn.Embedding(vocab_size, input_dim)
    embedding_model.load_state_dict(torch.load(base_path.joinpath(embedding_file)))
    model = TreeLSTMSentiment(
        cuda=False,
        vocab_size=vocab_size,
        in_dim=input_dim,
        mem_dim=mem_dim,
        num_classes=num_classes,
        model_name="constituency",
        criterion=None,
    )
    model.load_state_dict(torch.load(base_path.joinpath(model_file)))

    print(embedding_model)
    print(model)

    print("Done loading TreeLSTM model")
    return embedding_model, model, train_dataset, dev_dataset, test_dataset


if __name__ == '__main__':
    lt.monkey_patch()
    load_tree_lstm()
