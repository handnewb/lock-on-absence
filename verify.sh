#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════
#  verify.sh — run this after applying every change. Exit 0 means shippable.
#
#  This exists because three "fixes" were pushed in this project without the
#  program being run once: BodyDetector raised AttributeError in every code
#  path, enroll.py referenced two names that were never assigned, and a state
#  machine with green tests was never imported by production code. All three
#  are caught below in under a minute.
#
#  Usage:  ./verify.sh [/path/to/repo]
# ══════════════════════════════════════════════════════════════════════════
set -uo pipefail
REPO="${1:-.}"
cd "$REPO" || { echo "cannot cd to $REPO"; exit 1; }

PASS=0; FAIL=0
ok()   { printf "  \033[32m[ok]\033[0m   %s\n" "$*"; PASS=$((PASS+1)); }
bad()  { printf "  \033[31m[FAIL]\033[0m %s\n" "$*"; FAIL=$((FAIL+1)); }
step() { printf "\n\033[1m%s\033[0m\n" "$*"; }

VENV="${VENV:-/tmp/loa-verify-venv}"
step "0. clean venv install (catches broken pyproject / entry points)"
rm -rf "$VENV"
python3 -m venv "$VENV" >/dev/null 2>&1 || { bad "venv creation"; exit 1; }
PY="$VENV/bin/python"; BIN="$VENV/bin"
[ -x "$PY" ] || { PY="$VENV/Scripts/python.exe"; BIN="$VENV/Scripts"; }
if "$PY" -m pip install -q -e ".[dev]" 2>/dev/null; then ok "pip install -e '.[dev]'"
else bad "pip install -e '.[dev]'"; fi

step "1. static analysis"
if out=$("$PY" -m pyflakes lock_on_absence tests 2>&1) && [ -z "$out" ]; then
  ok "pyflakes: no undefined names, no dead imports"
else bad "pyflakes:"; echo "$out" | sed 's/^/        /'; fi
if "$PY" -m ruff check lock_on_absence tests -q >/dev/null 2>&1; then ok "ruff"
else bad "ruff"; "$PY" -m ruff check lock_on_absence tests --output-format=concise 2>&1 | head -10 | sed 's/^/        /'; fi

step "2. tests"
if out=$("$PY" -m pytest -q 2>&1); then
  ok "pytest: $(echo "$out" | grep -oE '[0-9]+ passed' | head -1)"
else bad "pytest"; echo "$out" | tail -25 | sed 's/^/        /'; fi

step "3. entry points resolve after install"
for c in lock-on-absence lock-on-absence-enroll lock-on-absence-watchdog lock-on-absence-replay; do
  if "$BIN/$c" --help >/dev/null 2>&1; then ok "$c --help"; else bad "$c --help"; fi
done

step "4. version is single-sourced"
V_INIT=$(grep -oP '__version__ = "\K[^"]+' lock_on_absence/__init__.py)
V_CLI=$("$BIN/lock-on-absence" --version 2>&1 | awk '{print $NF}')
V_README=$(grep -oP 'badge/version-\K[0-9]+\.[0-9]+\.[0-9]+' README.md | head -1)
[ "$V_INIT" = "$V_CLI" ] && ok "__init__ == CLI ($V_INIT)" || bad "__init__=$V_INIT CLI=$V_CLI"
[ "$V_INIT" = "$V_README" ] && ok "__init__ == README badge" || bad "__init__=$V_INIT README=$V_README"

step "5. backward-compatible shims (install.sh, install.ps1, systemd, README use these)"
for s in lock-on-absence.py enroll.py watchdog.py replay.py; do
  if [ -f "$s" ] && "$PY" "$s" --help >/dev/null 2>&1; then ok "$s"; else bad "$s"; fi
done

step "6. architectural guard: no presence logic in the adapter"
LEAK=0
for v in absence_start intruder_streak static_since prev_face_center \
         _body_detect_active last_face_time locked_until; do
  grep -q "$v" lock_on_absence/agent.py 2>/dev/null && { bad "legacy variable '$v' is back in agent.py"; LEAK=1; }
done
[ "$LEAK" -eq 0 ] && ok "agent.py holds no decision state"
grep -q "PresenceStateMachine" lock_on_absence/agent.py 2>/dev/null \
  && ok "state machine is on the production path" \
  || bad "agent.py does not use PresenceStateMachine"

step "7. FAR/FRR gates"
if "$BIN/lock-on-absence-replay" --synthetic --repeat 8 --far-window 12 \
     --fail-if-far-above 0.0 --fail-if-frr-above 0.0 >/dev/null 2>&1; then
  ok "clean scenario: FAR 0% / FRR 0%"
else bad "clean scenario gate"; fi
if "$BIN/lock-on-absence-replay" --synthetic --repeat 8 --flicker 0.4 --far-window 12 \
     --fail-if-far-above 0.15 --fail-if-frr-above 0.05 >/dev/null 2>&1; then
  ok "40% detection loss: within tolerance"
else bad "flicker gate"; fi
if ! "$BIN/lock-on-absence-replay" --synthetic --intruder-count 999 \
     --fail-if-far-above 0.0 >/dev/null 2>&1; then
  ok "the gate is able to fail (a gate that cannot fail is not a gate)"
else bad "gate did not fire on a deliberately broken config"; fi

step "8. governance files"
for f in LICENSE README.md SECURITY.md pyproject.toml .github/workflows/ci.yml; do
  [ -f "$f" ] && ok "$f" || bad "$f missing"
done

step "9. secrets and bulk in the git history"
if command -v git >/dev/null && [ -d .git ]; then
  SENS=$(git log --all --pretty=format: --name-only --diff-filter=A 2>/dev/null \
         | sort -u | grep -Ei '(^|/)\.env|\.pem$|\.key$|\.p12$|id_rsa|secrets?\.(ya?ml|json)' || true)
  [ -z "$SENS" ] && ok "no sensitive filenames in history" \
    || { bad "sensitive files in history:"; echo "$SENS" | sed 's/^/        /'; }
  PACK=$(git count-objects -vH | awk '/size-pack/{print $2, $3}')
  PACK_MB=$(git count-objects -v | awk '/size-pack/{printf "%.0f", $2/1024}')
  if [ "${PACK_MB:-0}" -gt 20 ]; then
    bad "pack is $PACK for $(du -sh --exclude=.git . 2>/dev/null | cut -f1) of code — every clone downloads it"
  else ok "pack size $PACK"; fi
  SIGNED=$(git log --all --format='%G?' 2>/dev/null | grep -c '^[GUX]' || true)
  TOTAL=$(git rev-list --all --count 2>/dev/null || echo 0)
  [ "${SIGNED:-0}" -gt 0 ] && ok "$SIGNED/$TOTAL commits signed" \
    || bad "0/$TOTAL commits signed — anyone can forge a commit as you"
fi

printf "\n════════════════════════════════════\n"
printf "  passed: %d   failed: %d\n" "$PASS" "$FAIL"
if [ "$FAIL" -gt 0 ]; then
  printf "  \033[31mNOT shippable\033[0m\n\n"; exit 1
fi
printf "  \033[32mshippable\033[0m\n\n"; exit 0
