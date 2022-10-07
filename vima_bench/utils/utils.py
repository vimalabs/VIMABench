import os
import tree
import numpy as np
import torch


def f_expand(fpath):
    return os.path.expandvars(os.path.expanduser(fpath))


def f_join(*fpaths):
    """
    join file paths and expand special symbols like `~` for home dir
    """
    return f_expand(os.path.join(*fpaths))


def stack_sequence_fields(sequence):
    """Stacks a list of identically nested objects.

    This takes a sequence of identically nested objects and returns a single
    nested object whose ith leaf is a stacked numpy array of the corresponding
    ith leaf from each element of the sequence.

    For example, if `sequence` is:

    ```python
    [{
          'action': np.array([1.0]),
          'observation': (np.array([0.0, 1.0, 2.0]),),
          'reward': 1.0
     }, {
          'action': np.array([0.5]),
          'observation': (np.array([1.0, 2.0, 3.0]),),
          'reward': 0.0
     }, {
          'action': np.array([0.3]),1
          'observation': (np.array([2.0, 3.0, 4.0]),),
          'reward': 0.5
     }]
    ```

    Then this function will return:

    ```python
    {
        'action': np.array([....])         # array shape = [3 x 1]
        'observation': (np.array([...]),)  # array shape = [3 x 3]
        'reward': np.array([...])          # array shape = [3]
    }
    ```

    Note that the 'observation' entry in the above example has two levels of
    nesting, i.e it is a tuple of arrays.

    Args:
      sequence: a list of identically nested objects.

    Returns:
      A nested object with numpy.

    Raises:
      ValueError: If `sequence` is an empty sequence.
    """
    # Handle empty input sequences.
    if not sequence:
        raise ValueError("Input sequence must not be empty")

    # Default to asarray when arrays don't have the same shape to be compatible
    # with old behaviour.
    try:
        return fast_map_structure(lambda *values: np.stack(values), *sequence)
    except ValueError:
        return fast_map_structure(lambda *values: np.asarray(values), *sequence)


def fast_map_structure(func, *structure):
    """Faster map_structure implementation which skips some error checking."""
    flat_structure = (tree.flatten(s) for s in structure)
    entries = zip(*flat_structure)
    # Arbitrarily choose one of the structures of the original sequence (the last)
    # to match the structure for the flattened sequence.
    return tree.unflatten_as(structure[-1], [func(*x) for x in entries])


def get_batch_size(x, strict: bool = False) -> int:
    """
    Args:
        x: can be any arbitrary nested structure of np array and torch tensor
        strict: True to check all batch sizes are the same
    """

    def _get_batch_size(x):
        if isinstance(x, np.ndarray):
            return x.shape[0]
        elif torch.is_tensor(x):
            return x.size(0)
        else:
            return len(x)

    xs = tree.flatten(x)

    if strict:
        batch_sizes = [_get_batch_size(x) for x in xs]
        assert all(
            b == batch_sizes[0] for b in batch_sizes
        ), f"batch sizes must all be the same in nested structure: {batch_sizes}"
        return batch_sizes[0]
    else:
        return _get_batch_size(xs[0])
