import math
from pathlib import Path
from random import choice, randint

import sympy
import torch
from sympy.abc import x
from tqdm import tqdm

from dynpartition.dataset.tree import Tree
from dynpartition.get_dir import get_saved_data_path

UNITARY_OPS = (
    sympy.sin,
    sympy.cos,
)
BINARY_OPS = (
    sympy.Add,
    sympy.Mul,
)

All_OPS = UNITARY_OPS + BINARY_OPS
STRING_BINARY_OPS = tuple([i.__name__ for i in BINARY_OPS])


def args(n, atoms, funcs):
    a = funcs + atoms
    g = []
    for _ in range(n):
        ai = choice(a)
        if isinstance(ai, sympy.FunctionClass):
            g.append(ai(*args(sympy.arity(ai), atoms, funcs)))
        else:
            g.append(ai)
    return g


def expr(ops, atoms=(-2, -1, 0, 1, 2)):
    while True:
        e = sympy.S.Zero
        while e.count_ops() < ops:
            t = choice(BINARY_OPS)(*args(randint(1, 3), atoms, UNITARY_OPS))
            e = choice(BINARY_OPS)(e, t)
            if e is sympy.S.NaN:
                break
        else:
            return e


def sympy_to_tree(equation):
    tree = Tree()

    if isinstance(equation, All_OPS):
        tree.layer = equation.func.__name__
        tree.name = str(equation)
        for arg in equation.args:
            child = sympy_to_tree(arg)

            if child is None:
                return None

            tree.add_child(child)
    else:
        tree.layer = 'Const'
        tree.name = equation

    solution = sympy.lambdify(x, equation)(1)
    if (
            isinstance(solution, complex)
            or abs(solution) > 10
            or math.isnan(solution)
            or math.isinf(solution)
    ):
        return None

    tree.gold_label = solution
    if not isinstance(equation, All_OPS):
        tree.value = tree.gold_label

    return tree


def get_proper_math_tree(num_ops):
    while True:
        tree = sympy_to_tree(expr(num_ops))
        if tree is None:
            continue

        if tree.depth() > num_ops:
            continue

        return tree


def create_pth_file(num_ops=5, dataset_size=10000):
    math_equations = []
    for _ in tqdm(
            range(dataset_size),
            desc=f'Generating {dataset_size} equations (num_ops={num_ops})',
            ascii=True,
            mininterval=1
    ):
        tree = get_proper_math_tree(num_ops)
        math_equations.append(tree.state_dict())

    folder = Path(__file__)
    while folder.name != 'dynpartition':
        folder = folder.parent
    torch.save(
        math_equations,
        get_saved_data_path().joinpath(f'math_equations_{num_ops}.pth')
    )
    return math_equations


if __name__ == '__main__':
    # max_eq = 100
    # max_ops = 10
    # c = 1
    # for i in range(max_eq):
    #     expression_tree = get_proper_math_tree(max_ops)
    #     i = str(i).zfill(math.ceil(math.log10(max_eq)) + 1)
    #     print(f'{i}: {expression_tree.gold_label:+2.4f} = {expression_tree}')

    for i in range(1, 11):
        create_pth_file(i, 10000)
