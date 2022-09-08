import collections.abc
import parafields._parafields as _parafields


def is_iterable(x):
    """Decide whether x is a non-string iterable"""
    if isinstance(x, str):
        return False
    return isinstance(x, collections.abc.Iterable)


def dict_to_parameter_tree(data, tree=_parafields.ParameterTree(), prefix=""):
    """Convert a (nested) dictionary to a C++ parameter tree structure"""
    for k, v in data.items():
        if isinstance(v, dict):
            dict_to_parameter_tree(v, tree=tree, prefix=prefix + k + ".")
        else:
            if is_iterable(v):
                v = " ".join([str(x) for x in v])
            tree.set(prefix + k, str(v))
    return tree


def generate(config={}):
    return _parafields.RandomField2D(dict_to_parameter_tree(config))
