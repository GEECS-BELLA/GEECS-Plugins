"""Client-side EPICS environment from the shared GEECS config.

The gateway host is infrastructure, like the MySQL database — clients should
resolve it from ``~/.config/geecs_python_api/config.ini`` rather than each
shell exporting ``EPICS_CA_ADDR_LIST``::

    [epics]
    ca_addr_list = 192.168.6.14
    # ca_auto_addr_list = NO      (optional; defaults to NO when
    #                              ca_addr_list is applied from here)
    [pva]
    file_plugin_addr_list = 192.168.6.100 192.168.7.161    # the camera servers
    # addr_list = ...              (the PVA image fleet; unioned in)
    # pva_auto_addr_list = NO      (optional; defaults to NO when the
    #                              list is applied from here)

The PVA list is the same rule for the same reason: the camera servers sit
on several subnets, so a PVA name search for the file plugin's PVs
(``…:hdf1:``) needs a directed address list — the worker's own
``[pva]`` keys already name those hosts, so nothing is exported twice
(found on the first plugin scan through the RE Manager, 2026-09-11: the
service environment carried the CA variables only and every plugin
signal timed out at connect).

Import-order constraint: libca reads these variables when the CA context is
created, which happens as soon as aioca is imported — and the device modules
import aioca (via ophyd-async) at package import.  ``geecs_bluesky/__init__``
therefore calls :func:`apply_epics_address_config` before importing any
submodule.  Explicitly exported environment variables always win
(``os.environ.setdefault`` semantics).
"""

from __future__ import annotations

import configparser
import logging
import os
from pathlib import Path
from typing import MutableMapping, Optional

logger = logging.getLogger(__name__)

_USER_CONFIG_PATH = Path("~/.config/geecs_python_api/config.ini")


def apply_epics_address_config(
    env: Optional[MutableMapping[str, str]] = None,
    config_path: Optional[Path] = None,
) -> dict[str, str]:
    """Apply ``[epics]`` config.ini values to the process environment.

    Sets ``EPICS_CA_ADDR_LIST`` from ``[epics] ca_addr_list`` when the
    variable is not already exported, and — only when the address list was
    applied from config — ``EPICS_CA_AUTO_ADDR_LIST`` from
    ``[epics] ca_auto_addr_list`` (default ``NO``: a directed address list
    plus broadcast is rarely intended).  Likewise ``EPICS_PVA_ADDR_LIST``
    from the union of ``[pva] file_plugin_addr_list`` and ``[pva]
    addr_list`` (in that order, duplicates dropped) and
    ``EPICS_PVA_AUTO_ADDR_LIST`` from ``[pva] pva_auto_addr_list``
    (default ``NO``).  Never raises: a missing file, section, or key is a
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
    path = (config_path or _USER_CONFIG_PATH).expanduser()

    applied: dict[str, str] = {}
    try:
        if not path.exists():
            return applied
        parser = configparser.ConfigParser()
        parser.read(path)
        addr = parser.get("epics", "ca_addr_list", fallback="").strip()
        if addr and "EPICS_CA_ADDR_LIST" not in env:
            env["EPICS_CA_ADDR_LIST"] = addr
            applied["EPICS_CA_ADDR_LIST"] = addr
            auto = parser.get("epics", "ca_auto_addr_list", fallback="NO").strip()
            if auto and "EPICS_CA_AUTO_ADDR_LIST" not in env:
                env["EPICS_CA_AUTO_ADDR_LIST"] = auto
                applied["EPICS_CA_AUTO_ADDR_LIST"] = auto
        pva_hosts: list[str] = []
        for key in ("file_plugin_addr_list", "addr_list"):
            for token in parser.get("pva", key, fallback="").replace(",", " ").split():
                if token not in pva_hosts:
                    pva_hosts.append(token)
        if pva_hosts and "EPICS_PVA_ADDR_LIST" not in env:
            env["EPICS_PVA_ADDR_LIST"] = " ".join(pva_hosts)
            applied["EPICS_PVA_ADDR_LIST"] = env["EPICS_PVA_ADDR_LIST"]
            auto = parser.get("pva", "pva_auto_addr_list", fallback="NO").strip()
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
        logger.warning("Could not apply [epics] config from %s", path, exc_info=True)
    return applied
