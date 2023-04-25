# tree object from stanfordnlp/treelstm
from __future__ import annotations

from typing import List, Optional, Tuple

from torch import Tensor


class Tree:
    def __init__(self):
        self.parent: Optional[Tree] = None
        self.num_children: int = 0
        self.children: List[Tree] = list()
        self.idx: Optional[int] = None  # node index for SST
        self.gold_label: Optional[int] = None  # node label for SST
        self.output: Optional[int] = None  # output node for SST
        self.state: Optional[Tuple[Tensor, Tensor]] = None

        # used by Math Functions only
        self.layer: Optional[str] = None  # layer of the node in the tree
        self.name: Optional[str] = None  # name of the node

    @property
    def value(self):
        return self.gold_label

    def add_child(self, child) -> Tree:
        child.parent = self
        self.num_children += 1
        self.children.append(child)
        return self

    def size(self) -> int:
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size
        return count

    def depth(self) -> int:
        count = 0
        if self.num_children > 0:
            for i in range(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth > count:
                    count = child_depth
            count += 1
        return count

    def __repr__(self):
        if self.layer is not None and self.name is not None:
            return f"{self.layer} : {self.name}"
        elif self.name is not None:
            return self.name
        elif self.layer is not None:
            return self.layer
        else:
            return super().__repr__()

    def is_leaf(self):
        return self.num_children == 0

    def get_leaf_nodes(self):
        if self.is_leaf():
            return [self]
        else:
            return sum([child.get_leaf_nodes() for child in self.children], [])

    def num_leaf_nodes(self):
        return len(self.get_leaf_nodes())

    def get_leaf_values(self, set_idx=False, offset=0):
        result = []
        for idx, leaf in enumerate(self.get_leaf_nodes()):
            if set_idx:
                leaf.idx = idx + offset

            result.append(leaf.gold_label)

        return result
