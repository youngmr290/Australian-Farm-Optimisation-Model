import pyomo.environ as pe
from itertools import product


def _as_tuple_list(pyomo_set):
    """
    Convert a Pyomo Set (possibly dimen=1 or >1) to a list of tuples.
    """
    res = []
    for el in pyomo_set:
        if isinstance(el, tuple):
            res.append(el)
        else:
            res.append((el,))
    return res


def build_active_set(
    base_index,
    *,
    prefix_sets=None,
    suffix_sets=None,
    drop_axes=None,
    order=None,
    doc=None,
):
    """
    Create a Pyomo Set on `model` called `name` by transforming a base index.

    Parameters
    ----------
    base_index : iterable[tuple]
        Base index tuples, typically from precalcs, e.g. params['idx_cropgraze_base_kp6p5zl']
        or a Pyomo Set (then we'll iterate over it).
    prefix_sets : list[pyomo Set] or None
        Pyomo sets whose elements should be prepended to each base tuple
        via cartesian product. E.g. [model.s_active_qs, model.s_feed_pools].
    suffix_sets : list[pyomo Set] or None
        Pyomo sets whose elements should be appended.
    drop_axes : iterable[int] or None
        Positions (0-based) in the *current* base tuple to drop.
        For example, with base (k,p6,p5,z,l), drop_axes=(4,) removes 'l'.
    order : iterable[int] or None
        Reordering of axes (after dropping). For example, if the tuple is
        currently (k,p6,p5,z) and you want (p6,k,p5,z) use order=(1,0,2,3).
    doc : str or None
        Docstring for the Pyomo Set.

    Returns
    -------
    Set
        The created Pyomo Set component.
    """

    # 0. Normalise base_index to a list of tuples
    if hasattr(base_index, "__iter__") and not isinstance(base_index, list):
        # could be a Pyomo Set, generator, etc.
        base_list = []
        for t in base_index:
            if isinstance(t, tuple):
                base_list.append(t)
            else:
                base_list.append((t,))
    else:
        base_list = list(base_index)

    # 1. Drop axes (on base) if requested
    if drop_axes is not None and base_list:
        drop_axes = set(drop_axes)
        kept_positions = [i for i in range(len(base_list[0])) if i not in drop_axes]
        projected = {tuple(t[i] for i in kept_positions) for t in base_list}
        base_list = sorted(projected)

    # 2. Prepare prefix and suffix lists (each element is a tuple)
    prefix_lists = []
    if prefix_sets is not None:
        for s in prefix_sets:
            prefix_lists.append(_as_tuple_list(s))

    suffix_lists = []
    if suffix_sets is not None:
        for s in suffix_sets:
            suffix_lists.append(_as_tuple_list(s))

    # 3. Build final tuples via cartesian product
    blocks = []
    if prefix_lists:
        blocks.extend(prefix_lists)
    blocks.append(base_list)
    if suffix_lists:
        blocks.extend(suffix_lists)

    final_index = []
    if blocks:
        for combo in product(*blocks):
            flat = ()
            for part in combo:
                flat += tuple(part)
            final_index.append(flat)

    # 4. Reorder axes (on FINAL tuple) if requested
    if order is not None and final_index:
        final_index = [tuple(t[i] for i in order) for t in final_index]

    dimen = len(final_index[0]) if final_index else 0
    return pe.Set(initialize=final_index, dimen=dimen, doc=doc)