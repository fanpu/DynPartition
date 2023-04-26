from pathlib import Path

import torch
from torch import nn

from dynpartition.dataset.generate_math_func import get_proper_math_tree
from dynpartition.dataset.sst_dataset import SSTDataset
from dynpartition.dataset.test import test_tree_lstm, test_math_model
from dynpartition.models.MathFuncSolver import MathFuncSolver
from dynpartition.models.TreeLSTM import TreeLSTMSentiment


def load_tree_lstm():
    num_classes = 3
    input_dim = 300
    mem_dim = 150
    data_folder = "saved_data"

    # vocab_file = "vocab-cased.pth"
    train_file = "sst_train_constituency_state_dict.pth"
    dev_file = "sst_dev_constituency_state_dict.pth"
    test_file = "sst_test_constituency_state_dict.pth"

    model_file = "20230425161219_constituency_model_state_dict_14.pth"
    embedding_file = "20230425161219_constituency_embedding_state_dict_14.pth"

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
    print("Embedding model loaded")

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
    print("Done loading TreeLSTM model")

    embedding_model.eval()
    model.eval()

    return embedding_model, model, train_dataset, dev_dataset, test_dataset


def load_math_model(dataset_size=1000, max_ops=5):
    model = MathFuncSolver()
    dataset = [get_proper_math_tree(max_ops) for _ in range(dataset_size)]
    return model, dataset


if __name__ == '__main__':
    device = torch.device("cuda" if (True and torch.cuda.is_available()) else "cpu")

    model, dataset = load_math_model()
    model.to(device)
    math_acc = test_math_model(device, model, dataset)
    print(f"Math accuracy: {math_acc * 100:.4f}%")

    embedding_model, model, train_dataset, dev_dataset, test_dataset = load_tree_lstm()
    model.to(device)
    embedding_model.to(device)
    dev_acc = test_tree_lstm(device, model, embedding_model, dev_dataset)
    print(f"Dev accuracy: {dev_acc * 100:.4f}%")
