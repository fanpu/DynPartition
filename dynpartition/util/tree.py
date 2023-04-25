# tree object from stanfordnlp/treelstm
from __future__ import annotations

from typing import List, Optional


class Tree:
    def __init__(self):
        self.parent: Optional[Tree] = None
        self.num_children: int = 0
        self.children: List[Tree] = list()
        self.idx: Optional[int] = None  # node index for SST
        self.gold_label: Optional[int] = None  # node label for SST
        self.output: Optional[int] = None  # output node for SST

        # used by Math Functions only
        self.layer = None  # layer of the node in the tree
        self.name: Optional[str] = None  # name of the node

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
        return f"{self.layer} : {self.name}"
