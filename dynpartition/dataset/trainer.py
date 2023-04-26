import sys

import torch
from tqdm import tqdm


class SentimentTrainer:
    """
    For Sentiment module
    """

    def __init__(self, cuda, model, embedding_model):
        super(SentimentTrainer, self).__init__()
        self.cuda = cuda
        self.model = model
        self.embedding_model = embedding_model

    @torch.no_grad()
    def test(self, dataset):
        self.model.eval()
        self.embedding_model.eval()
        predictions = torch.zeros(len(dataset))

        for idx in tqdm(range(len(dataset)), desc=f'Testing ', ascii=True, mininterval=1):
            tree, inputs, _ = dataset[idx]

            if self.cuda:
                inputs = inputs.cuda()

            emb = torch.unsqueeze(self.embedding_model(inputs), 1)
            output = self.model(tree, emb)  # size(1,5)

            output[:, 1] = -9999  # no need middle (neutral) value
            _, pred = torch.max(output, 1)
            predictions[idx] = pred

        sys.stdout.flush()
        sys.stderr.flush()
        return predictions
