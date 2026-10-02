"""GEECS Archiver Appliance — policies.py (site-neutral).

The appliance imports this file (Jython) to decide, per PV, where samples
go and how often. Our archive set is derived by `geecs-archiver onboard`
from the GEECS database; the request names a policy only when the
experiment's archive_policy.yaml overrides the default. Two stores:

  STS  short-term, hourly partitions, on the host path mounted at
       ARCHAPPL_SHORT_TERM_FOLDER (a tmpfs symlink is the upstream
       recommendation; a local SSD is fine)
  LTS  long-term, yearly partitions, on the mirrored data disk.

No MTS: at this facility's volume a two-rung ladder is enough.
Folders come from the environment so the file is the same on every host.
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
    """The policies the mgmt UI offers; keys are what archive_policy.yaml names."""
    return {
        "Default": "MONITOR, 1 s: every change, throttled to one stored sample per second",
        "Slow": "MONITOR, 10 s: slow-moving readbacks (vacuum, temperatures)",
        "Fast": "MONITOR, 0.1 s: the few PVs whose every 5 Hz update matters",
    }


def determinePolicy(pvInfoDict):
    """Pick the policy for one PV from its info dict."""
    name = pvInfoDict.get("policyName", "Default")
    period = {"Default": 1.0, "Slow": 10.0, "Fast": 0.1}.get(name, 1.0)
    return {
        "samplingPeriod": period,
        "samplingMethod": "MONITOR",
        "dataStores": STORES,
        "archiveFields": [],
    }


def getFieldsArchivedAsPartOfStream():
    """No EPICS record fields exist behind the gateway's PVs."""
    return []
