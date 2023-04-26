from __future__ import annotations

import dataclasses
from typing import List, Optional, Tuple, Union

import torch


@dataclasses.dataclass
class Tree:
    children: List[Tree] = dataclasses.field(default_factory=list)
    gold_label: Optional[int] = None  # node label
    value: Optional[int] = None  # node value for leaf nodes

    # used by Math Functions only
    layer: Optional[str] = None  # layer of the node in the tree
    name: Optional[str] = None  # name of the node in the tree

    # runtime
    parent: Optional[Tree] = None
    state: Union[None, Tuple[torch.Tensor, ...]] = None
    output: Optional[int] = None

    @property
    def num_children(self):
        return len(self.children)

    @property
    def label(self):
        return self.gold_label

    def state_dict(self):
        return {
            "children": [child.state_dict() for child in self.children],
            "gold_label": self.gold_label,
            "value": self.value,

            # used by Math Functions only
            "layer": self.layer,
            "name": self.name,
        }

    def load_state_dict(self, state_dict):
        self.children = [Tree().load_state_dict(child) for child in state_dict["children"]]
        for child in self.children:
            child.parent = self

        self.gold_label = state_dict["gold_label"]
        self.value = state_dict["value"]

        # used by Math Functions only
        self.layer = state_dict["layer"] if "layer" in state_dict else None
        self.name = state_dict["name"] if "name" in state_dict else None

        return self

    def add_child(self, child) -> Tree:
        child.parent = self
        self.children.append(child)
        return self

    def size(self) -> int:
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
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
            return f"Tree: label={self.label}, value={self.value}, children={self.num_children}"

    def is_leaf(self):
        return self.num_children == 0

    def get_leaf_nodes(self):
        if self.is_leaf():
            return [self]
        else:
            return sum([child.get_leaf_nodes() for child in self.children], [])

    def num_leaf_nodes(self):
        return len(self.get_leaf_nodes())
