#!/usr/bin/env bash
# Install repo git hooks. Run once after clone:  bash tools/install-hooks.sh
set -e
root="$(git rev-parse --show-toplevel)"
cp "$root/tools/pre-commit" "$root/.git/hooks/pre-commit"
chmod +x "$root/.git/hooks/pre-commit"
echo "installed: .git/hooks/pre-commit (size-gate, >5 MB blocked)"
