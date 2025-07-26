import datetime
import functools
import itertools
import pathlib
import sys
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence


def make_build_fn(pymodule_name: str, group_name: str):
    """Decorator for build functions.

    Automatically calls classes/functions in the decorated build function's module.
    """

    def decorator(wrapped_fn: Callable):
        @functools.wraps(wrapped_fn)
        def build_fn(
            config, name: Optional[str] = None, default_name: Optional[str] = None, **kwargs
        ):
            if config is None:
                raise ValueError(f"No config specified while building {group_name}")
            name = name or config.get("name") or default_name
            if name is None:
                raise ValueError(f"No name specified while building {group_name}")

            # Build modules with special handling
            module = wrapped_fn(config, name, **kwargs)

            # Build modules which do not need special handling
            if module is None:
                cls = get_class_by_name(pymodule_name, name)
                if cls is not None:
                    module = cls(**config_as_kwargs(config), **kwargs)
                else:
                    raise ValueError(f"Unknown {group_name} `{name}`")

            return module

        return build_fn

    return decorator


def get_class_by_name(module_name: str, name: str):
    pymodule = sys.modules[module_name]
    if name in pymodule.__dict__:
        return pymodule.__dict__[name]
    else:
        return None


def make_log_dir(
    log_dir: str, name: Optional[str] = None, group: Optional[str] = None
) -> pathlib.Path:
    time = datetime.datetime.now()
    dir_name = f"{time:%Y-%m-%d-%H-%M-%S}"
    if name is not None:
        dir_name += f"_{name}"

    if group is None:
        group = ""

    log_path = pathlib.Path(log_dir) / group / dir_name

    count = 2
    while log_path.is_dir():
        log_path = log_path.with_name(f"{dir_name}_{count}")
        count += 1

    log_path.mkdir(parents=True, exist_ok=True)
    return log_path


def config_as_kwargs(
    config, to_filter: Optional[Iterable[str]] = None, defaults: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Build kwargs for constructor from config dictionary."""
    always_filter = ("name",)
    if to_filter:
        to_filter = tuple(itertools.chain(always_filter, to_filter))
    else:
        to_filter = always_filter
    if defaults:
        # Defaults come first such that they can be overwritten by config
        to_iter = itertools.chain(defaults.items(), config.items())
    else:
        to_iter = config.items()
    return {k: v for k, v in to_iter if k not in to_filter}


def write_path(root: Any, path: str, value: Any):
    elems = path.split(".")
    parent = read_path(root, path, elems[:-1])
    elem = elems[-1]
    if isinstance(parent, Mapping):
        parent[elem] = value
    elif isinstance(parent, Sequence):
        try:
            index = int(elem)
        except ValueError:
            raise ValueError(
                f"Element {elem} of path `{path}` can not be converted into index "
                f"to index into sequence."
            ) from None
        parent[index] = value
    else:
        raise ValueError(
            f"Can not handle datatype {type(parent)} at element {elem} of path `{path}`"
        )


def read_path(
    root: Any, path: Optional[str] = None, elements: Optional[List[str]] = None, error: bool = True
):
    if path is not None and elements is None:
        elements = path.split(".")
    elif path is None and elements is None:
        raise ValueError("`elements` and `path` can not both be `None`")

    current = root
    for elem in elements:
        if isinstance(current, Mapping):
            next = current.get(elem)
            if next is None:
                if not error:
                    return None
                if path is None:
                    path = ".".join(elements)
                raise ValueError(
                    f"Can not use element {elem} of path `{path}` to access into "
                    f"dictionary. Available options are {', '.join(list(current))}"
                )
        elif isinstance(current, Sequence):
            try:
                index = int(elem)
            except ValueError:
                if not error:
                    return None
                if path is None:
                    path = ".".join(elements)
                raise ValueError(
                    f"Element {elem} of path `{path}` can not be converted to index into sequence."
                ) from None

            try:
                next = current[index]
            except IndexError:
                if not error:
                    return None
                if path is None:
                    path = ".".join(elements)
                raise ValueError(
                    f"Can not use element {elem} of path `{path}` to access into "
                    f"sequence of length {len(current)}"
                ) from None
        elif hasattr(current, elem):
            next = getattr(current, elem)
        else:
            if not error:
                return None
            if path is None:
                path = ".".join(elements)
            raise ValueError(
                f"Can not handle datatype {type(current)} at element " f"{elem} of path `{path}`"
            )

        current = next

    return current


def to_dict_recursive(dict_) -> dict:
    dict_ = {**dict_}
    for key, value in dict_.items():
        if isinstance(value, dict):
            dict_[key] = to_dict_recursive(value)
    return dict_
