from importlib.util import find_spec


def is_eflomal_available() -> bool:
    return find_spec("eflomal") is not None
