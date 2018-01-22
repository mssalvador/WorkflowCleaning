import numpy as np
import itertools


def draw(n, k, draws):
    sample = np.random.choice(range(n), replace=False, size=draws).tolist()
    return [k if i in sample else float('NAN') for i in range(n)]


def generate_string_data(z, a, k, draws=1):
    data = np.array([a * np.sin(z), a * np.cos(z), z]).T
    return list(zip(draw(len(z), float(k), draws), data))


def generate_springs(a, draws=1, *linspaces):
    tmp = [generate_string_data(z, a, idx, draws) for idx, z in enumerate(linspaces)]
    tmp3 = map(lambda x: (x[0], *x[1]), enumerate(itertools.chain(*tmp)))
    return list(tmp3)
