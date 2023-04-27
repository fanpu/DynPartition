import torch
import torch.nn as nn

from dynpartition.dataset.generate_math_func import STRING_BINARY_OPS, \
    get_proper_math_tree
from dynpartition.dataset.tree import Tree


class MathBinaryTreeLeafModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.one = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return (x * self.one,)


class MathBinaryTreeComposer(nn.Module):
    def __init__(self):
        super().__init__()
        self.one = nn.Parameter(torch.tensor(1.0))

    def forward(self, layer, *states):
        operation = getattr(torch, layer.lower(), None)

        if operation is None:
            raise ValueError(f"Operation {layer} not found in torch.")

        if layer in STRING_BINARY_OPS:
            if operation is torch.add:
                operation = torch.sum

            if operation is torch.mul:
                operation = torch.prod

            x = operation(torch.tensor(states, device=self.one.device))
        else:
            x = operation(*states)

        return (x * self.one,)


class MathCheckModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.one = nn.Parameter(torch.tensor(1.0))

    def forward(self, *x):
        return x[0] * self.one


class BinaryTreeMath(nn.Module):
    def __init__(self):
        super().__init__()
        self.leaf_module = MathBinaryTreeLeafModule()
        self.composer = MathBinaryTreeComposer()
        self.output_module = MathCheckModule()

    def forward(self, tree: Tree):
        if tree.num_children == 0:
            # Leaf Module
            tree.state = self.leaf_module.forward(tree.value)
        else:
            for child in tree.children:
                self.forward(child)

            # Non-leaf Module
            states = sum(
                [(tree.layer,)] + [child.state for child in tree.children], ()
            )
            tree.state = self.composer.forward(*states)

        # Output Module
        tree.output = self.output_module.forward(*tree.state)
        return tree.output


class MathFuncSolver(nn.Module):
    def __init__(self):
        super().__init__()
        self.tree_module = BinaryTreeMath()
        self.output_module = self.tree_module.output_module

    def forward(self, tree):
        return self.tree_module(tree)


if __name__ == '__main__':
    math_func_solver = MathFuncSolver()
    math_func_solver.eval()
    exp_tree = get_proper_math_tree(10)
    tree_inputs = torch.tensor(exp_tree.get_leaf_values(set_idx=True, offset=1))
    results = math_func_solver(exp_tree, tree_inputs)

    assert torch.isclose(
        torch.tensor(exp_tree.gold_label, dtype=torch.float),
        results
    )
