#!/usr/bin/env bash
#
# check.sh — run lint + unit tests the way CI does, scoped to what changed.
#
# Why this exists
# ---------------
# CI (.github/workflows/unit-tests.yml) runs each package's suite from a
# specific environment (root env vs the package's own env, with markers and
# env vars that differ per package). Re-deriving that mapping by hand every
# time is exactly the kind of tribal knowledge that rots; this script IS the
# mapping — when CI changes, change the runner table here in the same PR.
# The /check skill documents when to run which mode and how to read
# failures; /env-doctor fixes environment-shaped failures.
#
# Usage
# -----
#   ./scripts/check.sh                  # lint changed files + test changed packages
#   ./scripts/check.sh --lint           # lint only
#   ./scripts/check.sh --all            # lint everything + all locally runnable suites
#   ./scripts/check.sh GeecsBluesky …   # explicit package list (skips change detection)
#   ./scripts/check.sh --base <ref>     # diff against <ref> instead of auto-picking
#   ./scripts/check.sh --dry-run        # show the plan without running anything
#
# Change detection: tracked changes (committed, staged, unstaged) against the
# merge-base with the nearest integration branch (fewest commits since
# merge-base; candidates in pick_base mirror CONTRIBUTING.md § Branch
# topology); override with --base. Untracked files are deliberately
# invisible until `git add`ed — a scratch file inside a package must not
# trigger its whole suite.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# CI mapping (keep in sync with .github/workflows/unit-tests.yml):
#   root env  — root tests/ + these packages, marker "not integration and not gui"
#   own env   — these packages run their suite from inside the package dir
ROOT_ENV_PKGS="ImageAnalysis ScanAnalysis GEECS-Data-Utils GEECS-Schemas"
OWN_ENV_PKGS="GeecsBluesky GeecsCAGateway GEECS-Console GEECS-LogTriage"

MODE="changed"      # changed | all | lint
BASE=""
DRY_RUN=0
EXPLICIT_PKGS=""

usage() { sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
    case "$1" in
        --all)     MODE="all" ;;
        --lint)    MODE="lint" ;;
        --base)    BASE="${2:?--base needs a ref}"; shift ;;
        --dry-run) DRY_RUN=1 ;;
        -h|--help) usage; exit 0 ;;
        -*)        echo "check.sh: unknown option $1" >&2; usage >&2; exit 2 ;;
        *)         EXPLICIT_PKGS="$EXPLICIT_PKGS $1" ;;
    esac
    shift
done

contains() {  # contains "list of words" word
    case " $1 " in *" $2 "*) return 0 ;; *) return 1 ;; esac
}

pick_base() {
    # The named candidates mirror CONTRIBUTING.md § Branch topology (the
    # canonical copy) and die with it at M6. Deleted branches are skipped by
    # the rev-parse guard, so this degrades to origin/HEAD (the default
    # branch) with no edit required — pruning the stale names is cleanup,
    # not a correctness fix.
    best=""
    best_n=999999
    for c in origin/feat/vision-v1 origin/feat/greenfield-epics-bluesky-gui origin/HEAD origin/master; do
        git rev-parse --verify -q "$c" >/dev/null || continue
        mb="$(git merge-base HEAD "$c" 2>/dev/null)" || continue
        n="$(git rev-list --count "$mb..HEAD")"
        if [ "$n" -lt "$best_n" ]; then
            best_n="$n"
            best="$c"
        fi
    done
    echo "$best"
}

# --- Work out the changed-file set and the suites it implies -----------------
CHANGED=()          # changed files, NUL-delimited source (space-safe)
UNITS=""            # suites to run: "root-tests" and/or package names
UNCOVERED=""        # touched packages with no CI-mirrored suite

add_unit() { contains "$UNITS" "$1" || UNITS="$UNITS $1"; }

if [ -n "$EXPLICIT_PKGS" ]; then
    for p in $EXPLICIT_PKGS; do
        p="${p%/}"
        if contains "$ROOT_ENV_PKGS $OWN_ENV_PKGS" "$p" || [ "$p" = "root-tests" ]; then
            add_unit "$p"
        else
            echo "check.sh: '$p' is not a testable package (know: root-tests $ROOT_ENV_PKGS $OWN_ENV_PKGS)" >&2
            exit 2
        fi
    done
elif [ "$MODE" = "all" ]; then
    UNITS="root-tests $ROOT_ENV_PKGS"
    for p in $OWN_ENV_PKGS; do
        if [ -d "$p" ]; then
            UNITS="$UNITS $p"
        fi
    done
else
    [ -n "$BASE" ] || BASE="$(pick_base)"
    if [ -z "$BASE" ]; then
        echo "check.sh: no base branch found (no origin/feat/* or origin/master); use --base" >&2
        exit 2
    fi
    MB="$(git merge-base HEAD "$BASE")"
    while IFS= read -r -d '' f; do
        CHANGED+=("$f")
    done < <(git diff --name-only -z "$MB")
    for f in ${CHANGED[@]+"${CHANGED[@]}"}; do
        top="${f%%/*}"
        if [ "$top" = "tests" ]; then
            add_unit "root-tests"
        elif contains "$ROOT_ENV_PKGS $OWN_ENV_PKGS" "$top"; then
            if [ -d "$top" ]; then
                add_unit "$top"
            fi
        elif [ -f "$top/pyproject.toml" ]; then
            contains "$UNCOVERED" "$top" || UNCOVERED="$UNCOVERED $top"
        fi
        # everything else (scripts/, .github/, .claude/, docs/, Planning/…) → lint only
    done
fi

# --- Plan ---------------------------------------------------------------------
echo "== check.sh plan"
if [ -n "$EXPLICIT_PKGS" ]; then
    echo "   scope: explicit packages"
elif [ "$MODE" = "all" ]; then
    echo "   scope: --all (every locally runnable suite)"
else
    echo "   scope: changed vs $BASE (${#CHANGED[@]} tracked files; untracked ignored until git add)"
fi
echo "   lint : ${MODE}$( [ "$MODE" = "lint" ] && echo " (lint only)" )"
echo "   suites:$( [ "$MODE" = "lint" ] && echo " (none)" || echo "${UNITS:- (none — nothing testable changed)}" )"
for p in $UNCOVERED; do
    echo "   note : $p changed but has no CI-mirrored suite — test it manually if needed"
done

if [ "$DRY_RUN" -eq 1 ]; then
    echo "== dry run — nothing executed"
    exit 0
fi

# --- Lint (pre-commit; same fallback as scripts/commit.sh) ---------------------
if command -v pre-commit >/dev/null 2>&1; then
    PC=(pre-commit)
else
    PC=(poetry run pre-commit)
fi

LINT_OK=1
echo "== lint"
if [ -n "$EXPLICIT_PKGS" ]; then
    echo "   (explicit packages — lint skipped; run --lint or --all for it)"
elif [ "$MODE" = "all" ]; then
    "${PC[@]}" run --all-files || LINT_OK=0
elif [ "${#CHANGED[@]}" -gt 0 ]; then
    # Only lint files that still exist (deletions have nothing to lint).
    lint_files=()
    for f in ${CHANGED[@]+"${CHANGED[@]}"}; do
        if [ -e "$f" ]; then
            lint_files+=("$f")
        fi
    done
    if [ "${#lint_files[@]}" -gt 0 ]; then
        "${PC[@]}" run --files ${lint_files[@]+"${lint_files[@]}"} || LINT_OK=0
    else
        echo "   (nothing to lint)"
    fi
else
    echo "   (nothing to lint)"
fi

if [ "$MODE" = "lint" ]; then
    if [ "$LINT_OK" -eq 1 ]; then
        echo "== OK (lint)"
        exit 0
    fi
    echo "== FAILED (lint)"
    exit 1
fi

# --- Test suites ----------------------------------------------------------------
run_suite() {
    case "$1" in
        root-tests)
            # Root tests/ is mostly integration-marked; tolerate pytest's
            # no-tests-collected exit code (5), same as CI.
            poetry run pytest tests -m "not integration and not gui" --tb=short -q || [ $? -eq 5 ] ;;
        ImageAnalysis|ScanAnalysis|GEECS-Data-Utils|GEECS-Schemas)
            poetry run pytest "$1/tests" -m "not integration and not gui" --tb=short -q ;;
        GeecsBluesky)
            (cd GeecsBluesky && poetry run pytest tests -m "not integration and not fake_server" --tb=short -q) ;;
        GEECS-Console)
            (cd GEECS-Console && QT_QPA_PLATFORM=offscreen poetry run pytest --tb=short -q) ;;
        GeecsCAGateway|GEECS-LogTriage)
            (cd "$1" && poetry run pytest tests --tb=short -q) ;;
        *)
            echo "check.sh: no runner for '$1'" >&2; return 1 ;;
    esac
}

FAILED=""
for u in $UNITS; do
    if [ "$u" != "root-tests" ] && [ ! -d "$u" ]; then
        echo "== $u — not present on this branch, skipping"
        continue
    fi
    echo "== $u"
    run_suite "$u" || FAILED="$FAILED $u"
done

# --- Summary --------------------------------------------------------------------
echo ""
if [ "$LINT_OK" -eq 0 ] || [ -n "$FAILED" ]; then
    if [ "$LINT_OK" -eq 0 ]; then
        echo "== FAILED: lint"
    fi
    if [ -n "$FAILED" ]; then
        echo "== FAILED:$FAILED"
    fi
    echo "   (env-shaped failure — wrong python, missing extras, 'Command not found'," \
         "a layer collecting as '1 skipped'? → /env-doctor)"
    exit 1
fi
echo "== OK$( [ -n "$UNITS" ] && echo " —$UNITS" )"
