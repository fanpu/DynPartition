import math
import time
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
from torch import Tensor, nn

from dynpartition.dataset.load import load_math_model, load_tree_lstm
from dynpartition.dataset.tree import Tree
from dynpartition.get_dir import get_plot_path


class TreeNodesEncoding(nn.Module):
    def __init__(
            self,
            max_nodes: int = 200,
            dropout: float = 0,
            depth: Optional[int] = None
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_nodes = max_nodes

        self.depth = math.ceil(self.max_nodes / 4) if depth is None else depth
        self.depth = self.depth + 1 if self.depth % 2 == 1 else self.depth

        position = torch.arange(self.max_nodes).unsqueeze(1)
        position_encoding = torch.zeros(self.max_nodes, self.depth)

        div_term = torch.exp(
            torch.arange(0, self.depth, 2) * (-3.5 / self.depth)
        )

        position_encoding[:, 0::2] = torch.sin(position * div_term)
        position_encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('position_encoding', position_encoding)

    def plot_position_encoding(self, prefix: str = "") -> Path:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 5))
        plt.imshow(
            self.position_encoding,
            cmap='viridis',
            aspect='auto',
            extent=[0, self.depth, 0, self.max_nodes]
        )
        plt.xlabel('Depth')
        plt.xlim(0, self.depth)
        plt.ylabel('Position')
        plt.colorbar()
        plt.title('Positional Encoding')
        if prefix:
            prefix = f"{prefix}_" if prefix[-1] != "_" else prefix
        filepath = get_plot_path().joinpath(f"{prefix}positional_encoding.png")
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        return filepath

    def forward(self, x: Tensor, tree_size: Optional[int] = None) -> Tensor:
        assert x.dtype == torch.long
        te = self.position_encoding[x].sum(dim=-3) / x.shape[0]

        if tree_size is not None:
            mask = torch.arange(self.max_nodes) >= tree_size
            te[mask] = 0

        return self.dropout(te)


def encode_tree(
        tree: Tree,
        order: str = "in-order",
        max_nodes: Optional[int] = None,
        set_traversal_index: bool = False,
) -> np.ndarray:
    if order == "in-order":
        node_list = tree.in_order()
    elif order == "pre-order":
        node_list = tree.pre_order()
    elif order == "post-order":
        node_list = tree.post_order()
    else:
        raise ValueError(f"Invalid order {order}.")

    if set_traversal_index:
        tree.traversal_dict = {}
        for i, node in enumerate(node_list):
            node.traversal_index = i
            tree.traversal_dict[i] = node

    node_ids = [id(node) for node in node_list]
    position = list(range(1, 1 + len(node_list)))
    depth = [node.depth_from_root_parent() for node in node_list]
    parent = [
        (node_ids.index(id(node)) + 1 if node.parent is not None else 0)
        for node in node_list
    ]
    is_leaf = [int(node.is_leaf()) for node in node_list]

    matrix = np.array([position, depth, parent, is_leaf], dtype=np.int32)

    if max_nodes is not None and len(node_list) < max_nodes:
        matrix = np.pad(
            matrix,
            ((0, 0), (0, max_nodes - len(node_list))),
            mode="constant",
            constant_values=0
        )

    return matrix


def create_tree_embedding_dataset(
        trees: List[Tree],
        order: str = "in-order",
        max_num_nodes: Optional[int] = None,
        name: str = "",
        set_traversal_index: Optional[bool] = False,
        plot: bool = False
) -> List[Tensor]:
    if max_num_nodes is None:
        max_num_nodes = max([tree.size() for tree in trees]) + 1

    matrix_of_trees = [
        encode_tree(
            tree,
            order=order,
            max_nodes=max_num_nodes,
            set_traversal_index=set_traversal_index
        )
        for tree in trees
    ]
    matrix_of_trees = [
        torch.tensor(matrix, dtype=torch.long)
        for matrix in matrix_of_trees
    ]

    encoder = TreeNodesEncoding(max_nodes=max_num_nodes)
    if plot:
        encoder.plot_position_encoding(prefix=name)

    encoded_trees = [
        encoder(matrix, tree_size=tree.size())
        for tree, matrix in zip(trees, matrix_of_trees)
    ]
    return encoded_trees


if __name__ == '__main__':
    device = torch.device(
        "cuda" if (False and torch.cuda.is_available()) else "cpu"
    )
    _, dataset = load_math_model(device)
    _, train_dataset, dev_dataset, test_dataset = load_tree_lstm(device)

    start_time = time.time()
    print(encode_tree(dataset[0], order="in-order"))
    print(create_tree_embedding_dataset([dataset[0]])[0])
    _ = create_tree_embedding_dataset(
        dataset,
        set_traversal_index=True,
        name="math_model",
    )
    _ = create_tree_embedding_dataset(
        train_dataset.trees,
        set_traversal_index=True,
        name="train_sst",
    )
    _ = create_tree_embedding_dataset(
        dev_dataset.trees,
        set_traversal_index=True,
        name="dev_sst",
    )
    _ = create_tree_embedding_dataset(
        test_dataset.trees,
        set_traversal_index=True,
        name="test_sst",
    )
    print(f"Time: {time.time() - start_time:.2f}s")
