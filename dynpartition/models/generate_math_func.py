# TODO: Vivswan
import math
from random import choice, randint

import sympy
from sympy.abc import x

from dynpartition.util.tree import Tree

UNITARY_OPS = (
    sympy.sin,
    sympy.cos,
    # sympy.exp,
)
BINARY_OPS = (
    sympy.Add,
    sympy.Mul,
    # sympy.ln,
    # sympy.Pow,
    # sympy.atan,
)

All_OPS = UNITARY_OPS + BINARY_OPS


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
    return tree


if __name__ == '__main__':
    max_eq = 100
    max_ops = 10
    c = 1
    while c < max_eq + 1:
        expression_tree = sympy_to_tree(expr(max_ops))
        if expression_tree is None:
            continue

        if expression_tree.depth() > max_ops:
            continue

        c_str = str(c).zfill(math.ceil(math.log10(max_eq)) + 1)
        print(f'{c_str}: {expression_tree.gold_label:+2.4f} = {expression_tree}')
        c += 1
