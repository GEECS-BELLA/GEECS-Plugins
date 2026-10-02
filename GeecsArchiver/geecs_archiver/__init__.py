"""GEECS Archiver — keep an EPICS Archiver Appliance in step with a GEECS experiment.

The appliance itself is upstream's; it archives the CA gateway's PVs like any
other client.  This package is the glue a facility needs around it: the rule
that derives the archive set from the GEECS database (:mod:`archive_set`), a
typed client of the appliance's management API (:mod:`mgmt_client`), the
idempotent onboarding that reconciles the two (:mod:`onboard`), and the
``geecs-archiver`` command line (:mod:`cli`).  The deploy recipe lives in
``deploy/``; the operations runbook is ``DEPLOYMENT.md``.
"""

from geecs_archiver.archive_set import (
    ArchiveCandidate,
    Sampling,
    build_archive_set,
    derive_candidates,
    sampling_for,
)
from geecs_archiver.mgmt_client import MgmtClient, MgmtError, PVStatus
from geecs_archiver.onboard import OnboardPlan, VerifyReport, plan_onboarding, verify

__all__ = [
    "ArchiveCandidate",
    "Sampling",
    "build_archive_set",
    "derive_candidates",
    "sampling_for",
    "MgmtClient",
    "MgmtError",
    "PVStatus",
    "OnboardPlan",
    "VerifyReport",
    "plan_onboarding",
    "verify",
]
