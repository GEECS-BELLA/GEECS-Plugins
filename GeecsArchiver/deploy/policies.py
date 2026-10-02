"""GEECS Archiver Appliance — policies.py (site-neutral, one policy).

The appliance imports this file (Jython) to decide, per PV, where samples go
and how often. Our archive set is derived by `geecs-archiver onboard` from
the GEECS database, and every request carries its own sampling period and
method (the appliance's *user-specified sampling*, which takes precedence
over the period below) — so the experiment's archive_policy.yaml is the one
table and this file only names the stores:

  STS  short-term, hourly partitions, on the host path mounted at
       ARCHAPPL_SHORT_TERM_FOLDER (a tmpfs symlink is the upstream
       recommendation; a local SSD is fine)
  LTS  long-term, yearly partitions, on the mirrored data disk.

No MTS: at this facility's volume a two-rung ladder is enough. Folders come
from the environment so the file is the same on every host.
"""

import os

sts_root = os.environ["ARCHAPPL_SHORT_TERM_FOLDER"]
lts_root = os.environ["ARCHAPPL_LONG_TERM_FOLDER"]

STORES = [
    "pb://localhost?name=STS&rootFolder="
    + sts_root
    + "&partitionGranularity=PARTITION_HOUR&hold=2&gather=1",
    "pb://localhost?name=LTS&rootFolder="
    + lts_root
    + "&partitionGranularity=PARTITION_YEAR",
]


def getPolicyList():
    """The policies the mgmt UI offers — one; the request's own sampling sets the rate."""
    return {
        "Default": "STS (hourly) -> LTS (yearly); sampling as requested, else MONITOR 1 s"
    }


def determinePolicy(pvInfoDict):
    """The one policy: the stores, and a 1 s MONITOR fallback for a request that names no sampling."""
    return {
        "samplingPeriod": 1.0,
        "samplingMethod": "MONITOR",
        "dataStores": STORES,
        "archiveFields": [],
    }


def getFieldsArchivedAsPartOfStream():
    """No EPICS record fields exist behind the gateway's PVs."""
    return []
