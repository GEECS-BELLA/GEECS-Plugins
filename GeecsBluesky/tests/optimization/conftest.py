"""Skip the optional Xopt test layer when the extra is absent."""

import importlib.util

collect_ignore_glob = ["*"] if importlib.util.find_spec("xopt") is None else []
