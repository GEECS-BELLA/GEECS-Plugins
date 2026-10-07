"""Client-side EPICS environment from the shared GEECS config.

The gateway host is infrastructure, like the database: clients resolve it
from ``~/.config/geecs_python_api/config.ini`` instead of each shell
exporting ``EPICS_CA_ADDR_LIST``::

    [epics]
    ca_addr_list = <gateway host>
    # ca_auto_addr_list = NO     (default when ca_addr_list is applied here)
    [pva]
    file_plugin_addr_list = <the camera servers serving the file plugin>
    # addr_list = ...            (the PVA image fleet; unioned in)
    # pva_auto_addr_list = YES   (default: the directed list is added to
    #                             the broadcast search, so a local server
    #                             still resolves)

The PVA list follows the same rule: the camera servers sit on several
subnets, so the file plugin's PVs need a directed address list, and the
``[pva]`` keys already name those hosts.  The PVA auto list stays on by
default so a local server (a gateway against the fake GEECS server) is
not hidden.  On a service host the rendered ``config.ini`` carries both
keys from ``site.env``.

Import order: libca reads these variables when the CA context is created,
which happens as soon as aioca is imported, so ``geecs_bluesky/__init__``
calls :func:`apply_epics_address_config` before importing any submodule.
Explicitly exported environment variables always win (``setdefault``).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import MutableMapping, Optional

from geecs_core.configs_repo import CONFIG_PATH, read_config_entry

from geecs_bluesky.data_paths import pva_addr_tokens

logger = logging.getLogger(__name__)


def apply_epics_address_config(
    env: Optional[MutableMapping[str, str]] = None,
    config_path: Optional[Path] = None,
) -> dict[str, str]:
    """Apply the ``[epics]`` and ``[pva]`` config.ini addressing to the process environment.

    Sets ``EPICS_CA_ADDR_LIST`` from ``[epics] ca_addr_list`` when the
    variable is not already exported, and — only when the address list was
    applied from config — ``EPICS_CA_AUTO_ADDR_LIST`` from
    ``[epics] ca_auto_addr_list`` (default ``NO``: a directed address list
    plus broadcast is rarely intended).  Likewise ``EPICS_PVA_ADDR_LIST``
    from the union of ``[pva] file_plugin_addr_list`` and ``[pva]
    addr_list`` (in that order, duplicates dropped) and
    ``EPICS_PVA_AUTO_ADDR_LIST`` from ``[pva] pva_auto_addr_list``
    (default ``YES``: the list is added to the broadcast search, see the
    module docstring).  Never raises: a missing file, section, or key is a
    silent no-op so the env-var-only workflow keeps working unchanged.

    Parameters
    ----------
    env : MutableMapping[str, str], optional
        Environment mapping, injectable for testing (defaults to
        ``os.environ``).
    config_path : Path, optional
        Config file location, injectable for testing.

    Returns
    -------
    dict[str, str]
        The variables this call actually set (empty when nothing applied).
    """
    env = os.environ if env is None else env
    path = (config_path or CONFIG_PATH).expanduser()

    applied: dict[str, str] = {}
    try:
        if not path.exists():
            return applied
        addr = (read_config_entry("epics", "ca_addr_list", path) or "").strip()
        if addr and "EPICS_CA_ADDR_LIST" not in env:
            env["EPICS_CA_ADDR_LIST"] = addr
            applied["EPICS_CA_ADDR_LIST"] = addr
            auto = (
                read_config_entry("epics", "ca_auto_addr_list", path) or "NO"
            ).strip()
            if auto and "EPICS_CA_AUTO_ADDR_LIST" not in env:
                env["EPICS_CA_AUTO_ADDR_LIST"] = auto
                applied["EPICS_CA_AUTO_ADDR_LIST"] = auto
        pva_hosts = pva_addr_tokens(
            " ".join(
                read_config_entry("pva", key, path) or ""
                for key in ("file_plugin_addr_list", "addr_list")
            )
        )
        if pva_hosts and "EPICS_PVA_ADDR_LIST" not in env:
            env["EPICS_PVA_ADDR_LIST"] = " ".join(pva_hosts)
            applied["EPICS_PVA_ADDR_LIST"] = env["EPICS_PVA_ADDR_LIST"]
            auto = (
                read_config_entry("pva", "pva_auto_addr_list", path) or "YES"
            ).strip()
            if auto and "EPICS_PVA_AUTO_ADDR_LIST" not in env:
                env["EPICS_PVA_AUTO_ADDR_LIST"] = auto
                applied["EPICS_PVA_AUTO_ADDR_LIST"] = auto
        if applied:
            logger.info(
                "EPICS client addressing from %s: %s",
                path,
                ", ".join(f"{k}={v}" for k, v in applied.items()),
            )
    except Exception:
        logger.warning(
            "Could not apply [epics]/[pva] config from %s", path, exc_info=True
        )
    return applied
