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
  alone. An owned document read through a symbolic link (a file, a folder,
  or the experiment folder itself) is A's: a change to the link's target is
  judged as A's.
- **The log.** It records owned paths only, the same set the gate judged
  (links included). Another facility's commits get no row and never reach
  the apply seam.
- **The share.** The share's clone is a whole-repo checkout. It fast-forwards
  to the accepted main whoever moved main, including commits that touched
  other facilities only.

Each experiment is owned by exactly one box. No box can check what another
box claims, so the maintainer keeps the list (the `CODEOWNERS` below is the
same list for humans).

The boundary is the files, not what reads them. The analysis readers resolve
recipes by stem across the whole shared tree: two facilities' folders holding
the same stem fail the lookup, and a group may name another facility's
recipe. Another facility's commit to its own folder can still break this
box's analysis that way, and this box's gate does not stop it (it judges this
box's files only). Keep stems unique across facilities, and have each
facility's groups name its own recipes.

Symbolic links across facilities' folders are best avoided. A change that
reaches an owned document through a link is judged as the box's own, but a
**deleted** target is not seen: deletions are never judged, so the owned
document silently disappears from the box's listing.

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

   The scanner a dry run or an apply reaches (`GEECS_CONFIGURATOR_SCANNER_URL`,
   set at the switch below, `Exp=url,...` per experiment) is a separate key
   from ownership, and the two are never merged. An owned experiment normally
   has a scanner entry; without one, its rows stay `pending`. A scanner named
   for an experiment this box does not own is never reached by an apply,
   because the log has no rows for that experiment.
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

On the App's page, delete that box's private key. That box can no longer
mint a token, and the other boxes, each with its own key, are untouched. A
token already minted is **not** revoked with the key: the service reuses its
token for up to an hour, so for up to an hour the box (or whoever copied the
key and minted a token) can still push. For a stolen key, suspend the
App's installation on the configs repo until the hour has passed (or remove
the configs repo from the installation): that stops a token the thief
already minted, and it stops every box meanwhile. For a box merely retired,
stopping its service is enough: its token lives only in its memory. A box
whose key is deleted and whose service still runs answers 502 to saves, and
its inbox backs off, once its token expires. Then remove `/etc/geecs/configurator/github-app.pem` from the box. To
give the box a new key, repeat step 1. Nothing else on the box changes.

## Switching the configs root

The runbook for moving the Linux services' configs root from the share's
clone to the configurator's live checkout (ruling 4 on #1307), and for the
applies that switch turns on. Every step is the maintainer's: it needs
`sudo`, a unit restart, a `site.env` edit, or the maintainer's own git credential. Values
in angle brackets are the site's; nothing here names a host or an account.

After the switch:

- the Linux services (worker, scanner, portal, Canopy, Control Room, the
  archiver's CLI) read `<STATE_DIR>/live`, which holds **accepted commits of
  main only** and moves only by the configurator's fast-forward;
- the share's clone (`GEECS_CONFIGURATOR_SHARE_CLONE`) is fast-forwarded by
  the configurator after every accepted change, so LabVIEW keeps reading the
  file it always read; it is **read-only for humans and services**;
- the worker writes its one file (`shot_offsets.yaml`) into the inbox
  (`GEECS_CONFIGS_WRITE_ROOT`), and the configurator commits it;
- the scanner saves presets through the configurator (`--configurator-url`);
- the portal's analysis config editor is read-only until slice 6 moves the
  analysis kinds onto the configurator.

`<STATE_DIR>` is `GEECS_CONFIGURATOR_STATE_DIR` (the unit's
`StateDirectory`, `/var/lib/geecs-configurator`).

### Preconditions

1. The configurator runs with the commit path on (§ "Once per services box"
   above, its trial included); `/healthz` reads
   `"main": "applied"` and `"share": "current"`.
2. `GEECS_CONFIGURATOR_SHARE_CLONE` was set **before the configurator's first
   start**, so its live checkout started at the share's commit (else it
   started *unverified*: `/healthz` says so, and the maintainer looks at
   main and deletes `<STATE_DIR>/UNVERIFIED` before going on).
3. The lab is quiet: nothing runs or waits on the worker (the switch restarts
   the worker, the scanner, the portal and Canopy).
4. The trigger-profile catalogue is in the new schema, or the maintainer has
   said the resolver's legacy converter stays (#1307, their own items).

### 1. The share's edits, committed once (the maintainer's credential)

The share's clone may hold edits made by hand or by the old writers (the
scanner's presets, the worker's shot offsets, the portal's editor). They are
committed once, before anything fast-forwards it:

```
cd "<the share's configs clone>"
git status --short
git add <each file to keep>
git commit -m "Edits on the share before the configurator switch"
git push origin main
git status --short          # must print nothing
```

Then wait for the configurator's next fetch (`GEECS_CONFIGURATOR_FETCH_SECONDS`,
60 s by default) and check `/healthz`: `"main": "applied"`, `"share": "current"`.

### 2. site.env

`sudoedit /etc/geecs/site.env`, then set (keep each line's other flags):

```
GEECS_CONFIGURATOR_SHARE_CLONE=<the share's configs clone, the old GEECS_CONFIGS_ROOT>
GEECS_CONFIGS_ROOT=<STATE_DIR>/live
GEECS_CONFIGS_WRITE_ROOT=<STATE_DIR>/inbox
GEECS_CONFIGURATOR_SCANNER_URL=<experiment>=http://127.0.0.1:<its scanner's port>,<instance experiment>=http://127.0.0.1:<its scanner's port>
GEECS_CONFIGURATOR_ARCHIVER_COMMAND=<checkout root>/archiver-checkout/GeecsArchiver/.venv/bin/geecs-archiver
GEECS_SCANNER_EXTRA_ARGS=<as before> --configurator-url http://127.0.0.1:<GEECS_CONFIGURATOR_PORT>
GEECS_PORTAL_EXTRA_ARGS=<as before> --config-editor-read-only
```

`GEECS_CONFIGURATOR_SCANNER_URL` names each experiment's scanner (an instance's
scanner is on its slot's port, `deploy/experiment.sh list`): a document of an
experiment with no scanner named is never applied by another's reopen; its row
stays pending, named. One plain URL is the served experiment's alone.

The scanner's preset panel embeds the editor (Edit preset…) and the Control
Room links documents into it: both need the configurator's URL **as the browser
reaches it**, and the scanner's origin in the configurator's CORS list:

```
GEECS_SCANNER_EXTRA_ARGS=<as before> --configurator-browser-url http://<this host>:<GEECS_CONFIGURATOR_PORT>
GEECS_CONFIGURATOR_CORS_ORIGINS=http://<this host>:<the scanner's port>
GEECS_CONTROL_ROOM_CONFIG_EDITOR_URL=http://<this host>:<GEECS_CONFIGURATOR_PORT>
```

Unset, the scanner shows no Edit preset… and the Control Room no Edit links.

The worker, the scanner and the configurator run as the one service account,
which reads `<STATE_DIR>` (mode 0755) and writes the inbox (the configurator
creates each owned experiment's folder there at its start; a new experiment needs
a configurator restart).

### 3. The rendered config.ini (by hand)

The bootstrap writes the service account's
`~/.config/geecs_python_api/config.ini` only when it is absent, so edit it:

```
[Paths]
scanner_config_root_path = <STATE_DIR>/live
scan_analysis_configs_path = <STATE_DIR>/live/scan_analysis_configs
```

### 4. Units and Canopy's compose, re-rendered

```
cd <checkout root>/qs-checkout
deploy/render_units.sh /etc/geecs/site.env ~/deploy-staging
GeecsCanopy/deploy/render_conf.sh /etc/geecs/site.env ~/deploy-staging/canopy
```

Install with the `sudo` lines each prints. Canopy's compose now mounts the
live checkout's own git dir (`<STATE_DIR>/repo.git/worktrees/live`) read-only
at the same path, so its readers see a fast-forward's rewrite lock and never
read the device registry or the operator policy as missing mid-rewrite.

### 5. Restart, in order

```
sudo systemctl restart geecs-configurator
sudo systemctl restart geecs-qserver geecs-qserver-ready geecs-scanner geecs-data-portal geecs-logbook geecs-canopy geecs-control-room
```

(Experiment instances: their `@<Instance>` units too, the logbook's included.)

### 6. Check

- `curl -s http://127.0.0.1:<GEECS_CONFIGURATOR_PORT>/healthz` →
  `"main": "applied"`, `"share": "current"`.
- The scanner's `/api/status` → `readiness: "ready"`.
- `git -C <STATE_DIR>/live status --short` prints nothing, now and after a
  scan.
- The portal's `/configs/` page says read-only; a save there answers 405.

### The acceptance this switch enables (OWED, on a bench experiment instance)

Laser off, nothing of the instance running or queued; never a production
experiment's device. From a browser on the configurator's page:

1. Open the laser-off trigger profile, change one amplitude, enter the
   operator's name, Save. Expected: the log (`/log`) shows a `saved` row by
   "<name> via configurator", state **queued**; `git -C <the share's configs
   clone> log -1` shows the same commit within seconds.
2. Within `GEECS_CONFIGURATOR_APPLY_SECONDS` (30 s): the scanner's chip reads
   "reopening", then ready; the row reads **applied** ("the worker
   environment reopened (… s)").
3. Run the laser-off scan. Expected: the trigger box fires at the new
   amplitude (its `:SP` readback in the scan's s-file, or a `caget` of the
   gateway PV).
4. The refusal: start a scan, then save another amplitude change. Expected:
   the row stays **queued** ("waits for the worker idle and empty") until the
   scan ends, then applies.

### After the switch

- **Nothing writes the live checkout.** An untracked or edited file there
  stops its every fast-forward (`/healthz` and `/log` say so) until it is
  removed by hand. A new experiment's folder goes into the configs repo by
  pull request; then restart `geecs-configurator` (it creates the inbox
  folder at start). `deploy/experiment.sh up` refuses to scaffold into the
  live checkout and says so.
- **A save during an apply pass is answered busy** (503, `Retry-After: 60`):
  the pass holds the repository while the worker reopens (a minute or
  more). Nothing was saved; save again. The scanner's Save as preset shows
  the same message.

### Rollback

Put back `GEECS_CONFIGS_ROOT` (the share's clone), empty
`GEECS_CONFIGS_WRITE_ROOT`, drop `--configurator-url` and
`--config-editor-read-only`, restore `config.ini`'s two paths, re-render and
restart as in steps 4 and 5. The share's clone is current with main, so
nothing is lost; presets then cannot be saved from the scanner until the
switch is redone.
