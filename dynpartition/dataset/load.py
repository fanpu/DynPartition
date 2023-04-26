import warnings

import torch
from torch import nn

from dynpartition.dataset.generate_math_func import create_pth_file
from dynpartition.dataset.sst_dataset import SSTDataset
from dynpartition.dataset.tree import Tree
from dynpartition.get_dir import get_saved_data_path
from dynpartition.models.MathFuncSolver import MathFuncSolver
from dynpartition.models.TreeLSTM import TreeLSTMSentiment


def load_tree_lstm(device):
    num_classes = 3
    input_dim = 300
    mem_dim = 150

    # vocab_file = "vocab-cased.pth"
    train_file = "sst_train_constituency_state_dict.pth"
    dev_file = "sst_dev_constituency_state_dict.pth"
    test_file = "sst_test_constituency_state_dict.pth"

    model_file = "20230425161219_constituency_model_state_dict_14.pth"
    embedding_file = "20230425161219_constituency_embedding_state_dict_14.pth"

    data_path = get_saved_data_path()
    print("Loading TreeLSTM model...")

    vocab_size = 21699
    # vocab_size = Vocab().load_state_dict(torch.load(data_path.joinpath(vocab_file))).size()
    dev_dataset = SSTDataset().load_state_dict(
        torch.load(data_path.joinpath(dev_file), map_location=device)
    )
    train_dataset = SSTDataset().load_state_dict(
        torch.load(data_path.joinpath(train_file), map_location=device)
    )
    test_dataset = SSTDataset().load_state_dict(
        torch.load(data_path.joinpath(test_file), map_location=device)
    )

    embedding_model = nn.Embedding(vocab_size, input_dim)
    embedding_model.load_state_dict(
        torch.load(data_path.joinpath(embedding_file), map_location=device)
    )
    print("Embedding model loaded")

    model = TreeLSTMSentiment(
        cuda=False,
        vocab_size=vocab_size,
        in_dim=input_dim,
        mem_dim=mem_dim,
        num_classes=num_classes,
        model_name="constituency",
        embedding_model=None,
    )
    model.load_state_dict(
        torch.load(data_path.joinpath(model_file), map_location=device),
        strict=False,
    )
    print("Model loaded")
    model.tree_module.embedding_model = embedding_model

    embedding_model.eval()
    model.eval()

    return model, train_dataset, dev_dataset, test_dataset


def load_math_model(device, max_ops=5, dataset_size=10000):
    dataset_file = f"math_equations_{max_ops}.pth"

    data_path = get_saved_data_path()
    if not data_path.joinpath(dataset_file).exists():
        create_pth_file(max_ops, dataset_size)

    dataset = torch.load(data_path.joinpath(dataset_file), map_location=device)

    if dataset_size > len(dataset):
        warnings.warn(
            f"Requested dataset size ({dataset_size}) is larger than cached dataset ({len(dataset)})")
    else:
        dataset = dataset[:dataset_size]

    dataset = [Tree().load_state_dict(tree) for tree in dataset]

    model = MathFuncSolver()
    model.eval()

    return model, dataset
