---
name: lab-day
description: >
  Prepare or close out a lab session against the hardware-verification
  ledger in PR bodies. Use before lab time ("what's owed on hardware",
  "lab day checklist", "what should I verify next time I'm at the lab /
  on the lab network") to sweep open and recently merged PRs for OWED
  items and build a run list with expected numbers; use after the session
  ("here are the lab results", "close out the lab day") to turn results
  into PR verification comments and flip OWED to VERIFIED.
---

# /lab-day — hardware-verification ledger

The PR template requires a **Hardware verification** section (VERIFIED /
OWED / N-A), which makes PR bodies a distributed ledger of everything
code-complete but not yet hardware-verified. This skill cashes that
ledger in: **prep mode** builds the checklist before a lab session,
**closeout mode** writes the results back afterwards.

`$ARGUMENTS`: empty or "prep" → prep mode; results, notes, or "closeout"
→ closeout mode.

## Prep mode — build the run list

1. **Sweep the ledger.** Collect OWED items from:
   - Open PRs: `gh pr list --state open --json number,title,body`
   - Recently merged PRs (~45 days):
     `gh pr list --state merged --limit 40 --json number,title,body,mergedAt`
   - Match case-insensitively on `owed` within the Hardware verification
     section; for any PR whose body has that section, also skim
     `gh pr view <n> --comments` — verification often lands as a
     follow-up comment, and a comment saying VERIFIED retires the body's
     OWED entry.
   - `grep -ri "owed" Planning/ --include="*.md"` — design notes carry
     verification plans too (e.g. the read-path phase notes).
2. **Build the checklist.** One entry per still-open item:
   - What to run (exact acceptance test — command, scan shape, rig)
   - Expected result (the numbers the PR promised)
   - Source (PR #N or planning doc)
   - Preconditions (lab network, which machine, operator coordination —
     e.g. anything touching the DG645 gates the gas jet; scripts with an
     operator-awareness header like
     `GeecsBluesky/scripts/verify_staging_live.py` mean exactly that)
3. **Order it.** Group by rig/setup so reconfiguration happens once;
   put items blocking open PR merges first, merged-PR confirmations
   second, nice-to-haves last. Flag anything that needs a second person.
4. Present the checklist. Offline is fine for prep — never ping the lab
   DB or devices while building it.

## Closeout mode — write results back

1. Match each reported result to its ledger entry; compute pass/fail
   against the expected numbers.
2. For each item, draft the verification comment for its PR: what was
   run, where, the measured numbers vs expected, verdict. Show all
   drafts to the user and get explicit confirmation **before posting
   anything** (`gh pr comment <n> --body …`).
3. For still-open PRs, also offer to edit the body's Hardware
   verification section from OWED to VERIFIED (with the numbers), so the
   ledger stays accurate at merge time.
4. A result that *contradicts* the expectation is a finding, not a
   bookkeeping entry: keep the OWED status, summarize the discrepancy,
   and offer to file an issue (evidence-first, per the bug template).

## Notes

- The ledger's unit is the PR, deliberately — no separate tracking file
  to rot. If an owed item has no PR (e.g. verbal/pre-template debt),
  suggest recording it as a checklist comment on the nearest related PR
  or an issue, so the next sweep sees it.
