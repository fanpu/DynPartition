from pathlib import Path

import torch
from torch import nn

from dynpartition.dataset.accuracy import sentiment_accuracy_score
from dynpartition.dataset.sst_dataset import SSTDataset
from dynpartition.dataset.trainer import SentimentTrainer
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


if __name__ == '__main__':
    cuda = True and torch.cuda.is_available()
    embedding_model, model, train_dataset, dev_dataset, test_dataset = load_tree_lstm()

    if cuda:
        model.cuda()
        embedding_model.cuda()

    trainer = SentimentTrainer(cuda, model, embedding_model)
    dev_pred = trainer.test(dev_dataset)
    dev_acc = sentiment_accuracy_score(dev_pred, dev_dataset.labels)
    print(f"Dev accuracy: {dev_acc * 100:.4f}%")
