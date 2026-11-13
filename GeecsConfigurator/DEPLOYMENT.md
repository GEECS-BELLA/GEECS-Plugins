# Deploying GeecsConfigurator's commit path

One configs repository serves several facilities. Each facility has its own
services box, and each box runs one configurator that commits for that
facility alone. They share one GitHub App, installed once, with a private key
per box. This page is generic: every value in angle brackets is your own, and
the facility values live in each box's `/etc/geecs/site.env`
(`deploy/site.env.example` lists every key).

The read-only configurator (no `GEECS_CONFIGURATOR_STATE_DIR`) needs none of
this. `deploy/bootstrap_host.sh --only configurator` installs it.

## The rule: a box owns its experiments

A box commits, judges and logs only the paths it **owns**
(`geecs_configurator/ownership.py`):

- `scanner_configs/experiments/<name>/` for each `<name>` in
  `GEECS_CONFIGURATOR_EXPERIMENTS`;
- `scan_analysis_configs/<kind folder>/<name>/` (the analyzers and the groups)
  for each `<name>` in `GEECS_CONFIGURATOR_ANALYSIS_FOLDERS`. The analysis
  tree is shared across facilities and keeps one folder per facility, so a box
  is given its own folders there by name. It never gets the whole tree. A file
  at the top of a kind folder, outside any facility's folder, belongs to no
  box: it changes by pull request.

With the commit path on, at least one of the two keys must name something. A
box that owns nothing refuses to start, because an empty setting is never read
as "everything".

What follows from that:

- **Writes.** A save, a maintainer-tier proposal or an inbox publish outside
  the owned paths is refused with a 403, before anything is staged. That
  includes a path through a symbolic link that leaves an owned folder. Every
  experiment can still be browsed: another facility's documents are served
  read-only, and the page says whose they are.
- **The gate.** Before the live checkout fast-forwards, it judges only the
  owned documents a commit changed. Another facility's commit fast-forwards
  without judgement, valid or not: that facility's box and the configs repo's
  `validate` check judge it. So facility B's invalid commit never blocks
  facility A's saves, while A's own invalid commit still blocks A until main is
  fixed by pull request. A commit that touches both is judged on A's paths
  alone.
- **The log.** It records owned paths only. Another facility's commits get no
  row and never reach the apply seam: they are that facility's box's business.
- **The share.** The share's clone is a whole-repo checkout. It fast-forwards
  to the accepted main whoever moved main, including commits that touched
  other facilities only.

Each experiment is owned by exactly one box. No box can check what another
box claims, so the maintainer keeps the list (the `CODEOWNERS` below is the
same list for humans).

The App is one identity for every box, so GitHub cannot stop a box from
pushing outside its folders. That limit is enforced on each box, from its
site.env. A box's key is its whole authority: guard it like a deploy key and
revoke it the same way (below).

## Once per org

1. **Create the GitHub App** (org settings, Developer settings, GitHub Apps,
   New GitHub App):
   - name: `<org>-configurator` (any name; GitHub shows it as the pusher);
   - homepage URL: any, e.g. the configs repo's URL;
   - **Webhook: off** (uncheck *Active*). The boxes fetch on a timer;
   - repository permissions: **Contents: Read and write**, **Pull requests:
     Read and write**, **Metadata: Read-only** (GitHub requires it). Nothing
     else, and no organization or account permissions;
   - where it can be installed: *Only on this account*.

   Note the **App ID** on the App's page.
2. **Install it on the configs repository alone** (the App's page, Install
   App, the org, *Only select repositories*, the configs repo). The
   installation's URL ends in the **installation ID**
   (`.../settings/installations/<installation id>`).
3. **The ruleset on `main`** (the configs repo's settings, Rules, Rulesets,
   New branch ruleset, target the default branch):
   - *Require a pull request before merging*, plus *Require review from Code
     Owners* once `CODEOWNERS` is in;
   - *Block force pushes*;
   - *Restrict deletions*;
   - *Require status checks to pass*: `validate`, added once the workflow has
     run once (step 4) so GitHub knows the check's name;
   - **bypass list**: the App, plus the maintainers as a repository role
     (*Repository admin*) or a team. Rulesets name roles, teams and apps, not
     users. The App's bypass is the whole operator tier; the maintainers'
     bypass is how a regenerated catalogue is committed directly.
4. **Commit the check and the owners** to the configs repo's main (as a
   maintainer):
   - `deploy/configs-repo/validate.yml` from this package, unchanged, as
     `.github/workflows/validate.yml`. The check judges every changed document
     on every push and pull request, whichever facility it belongs to;
   - `deploy/configs-repo/CODEOWNERS.example`, filled in, as
     `.github/CODEOWNERS`: each facility's folders to that facility's
     maintainers.

## Once per services box

As root on the box, with `<service account>` the account the units run as.

1. **The box's own private key.** On the App's page, *Generate a private key*.
   Each box gets its own key, so one box can be revoked alone. Copy the
   downloaded `.pem` to the box, install it, then delete every other copy:

   ```bash
   sudo install -d -m 0750 -o <service account> /etc/geecs/configurator
   sudo install -m 0600 -o <service account> <downloaded>.pem /etc/geecs/configurator/github-app.pem
   ```
2. **The site.env keys** (`/etc/geecs/site.env`; the example values are
   placeholders):

   ```bash
   GEECS_CONFIGURATOR_STATE_DIR=/var/lib/geecs-configurator
   GEECS_CONFIGURATOR_ORIGIN=https://github.com/<org>/<configs repo>.git
   GEECS_CONFIGURATOR_GITHUB_REPO=<org>/<configs repo>
   GEECS_CONFIGURATOR_GITHUB_APP_ID=<app id>
   GEECS_CONFIGURATOR_GITHUB_INSTALLATION_ID=<installation id>
   GEECS_CONFIGURATOR_GITHUB_APP_KEY_FILE=/etc/geecs/configurator/github-app.pem
   # what this box owns: its experiments, its folders in the shared analysis tree
   GEECS_CONFIGURATOR_EXPERIMENTS=<ExperimentA1>,<ExperimentA2>
   GEECS_CONFIGURATOR_ANALYSIS_FOLDERS=<FacilityA>
   # a committer per box, so the log and git history say which box committed
   GEECS_CONFIGURATOR_COMMITTER_NAME=GEECS configurator (<facility>)
   GEECS_CONFIGURATOR_COMMITTER_EMAIL=configurator+<facility>@<domain>
   GEECS_CONFIGURATOR_INBOX_AUTHOR=GEECS worker (<facility>) via configurator
   GEECS_CONFIGURATOR_SHARE_CLONE=<the share's clone of the configs repo>
   ```

   The committer e-mail must differ per box. A box treats commits under its
   own e-mail as its own, and never logs or applies them again.

   The scanner a dry run or an apply reaches (slice 3's
   `GEECS_CONFIGURATOR_SCANNER_URL`, which may name one per experiment) is a
   separate key from ownership, and the two are never merged. An owned
   experiment normally has a scanner entry: without one, its rows stay
   `pending`.
3. **The state directory** is the unit's `StateDirectory`
   (`/var/lib/geecs-configurator`, mode 0755: the live checkout in it is the
   configs root the box's Linux services read). systemd creates it. Nothing
   else writes there.
4. **The share's clone** must be on `main`, with no edits, and must be a
   commit main contains: the service starts from it, so main's newer commits
   pass the gate. When the share's files belong to another owner, the service
   account needs
   `git config --global --add safe.directory <the share's clone>`. Set
   `GEECS_CONFIGURATOR_SHARE_CLONE` **before the first start**. Without it the
   service starts *unverified* (see `/healthz`), and holds the share until the
   maintainer deletes `<STATE_DIR>/UNVERIFIED`.
5. **Render, start, check:**

   ```bash
   deploy/render_units.sh            # or deploy/bootstrap_host.sh --only configurator
   sudo systemctl restart geecs-configurator
   curl -s http://127.0.0.1:<port>/healthz
   ```

   `/healthz` should show `"commits": "on"`, `"credential": "GitHub App"`,
   `"main": "applied"`, `"share": "current"`, and `"owns"` listing this box's
   experiments and analysis folders. A box that owns nothing does not start:
   `journalctl -u geecs-configurator` names the missing key.
6. **The trial, on a throwaway branch**, before the box commits to main: set
   `GEECS_CONFIGURATOR_BRANCH=<trial branch>` (made from main on GitHub, and
   left out of the ruleset), restart, then:
   - save an operator-tier document in an owned experiment. It lands on the
     branch, authored by the operator, and shows on `/log`;
   - save a maintainer-tier one. A pull request opens against the branch;
   - open another facility's document. It is read-only, and a save is a 403;
   - push an invalid document to another facility's folder on the branch. The
     box fast-forwards past it, and `/log` has no row for it;
   - push an invalid document to an owned folder. The box refuses it, `/log`
     shows `refused`, and a save answers 502 until the branch is fixed.

   Then unset `GEECS_CONFIGURATOR_BRANCH`, delete the trial branch, empty
   `<STATE_DIR>` (it followed the trial branch), and restart.

### Revoking a box's key

On the App's page, delete that box's private key. That box can no longer mint
a token: its saves answer 502 and its inbox backs off. The other boxes, each
with its own key, are untouched. Then remove
`/etc/geecs/configurator/github-app.pem` from the box. To give the box a new
key, repeat step 1. Nothing else on the box changes.

## Switching the configs root

The runbook for this lands with slice 3b.
