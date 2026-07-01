"""Test configuration for GEECS-Scanner-GUI.

Patches the GEECS-PythonAPI MySQL database connection at pytest configuration
time (before test collection) so that imports of scan_device.py and other
modules that call GeecsDatabase at module level succeed without a lab network.

``scan_device.py`` runs ``GeecsDatabase.collect_exp_info(expt)`` at import time,
guarded only against ``AttributeError`` — not against a MySQL connection
timeout. On CI there is no ``config.ini`` so ``load_config()`` returns ``None``
and that path is skipped, but a developer machine that *has* a ``config.ini``
and is off the lab network would otherwise block ~75 s and then fail to import
the engine package (and anything that pulls it in, like the optimization
wrapper). Starting the patch here — at conftest import, before any test module
is collected — keeps those imports network-free.
"""

from unittest.mock import patch

_collect_exp_info_patch = patch(
    "geecs_python_api.controls.interface.GeecsDatabase.collect_exp_info",
    return_value={"devices": {}},
)
_collect_exp_info_patch.start()
