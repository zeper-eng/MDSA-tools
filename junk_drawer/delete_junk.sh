cd "/c/Users/Luis/Projects/MDSA_tools_revisions/MDSA-tools" || exit 1

set -euo pipefail

# ==========================================================
# 1. Verify repository
# ==========================================================

git rev-parse --is-inside-work-tree
git branch --show-current

if [[ "$(git branch --show-current)" != "main" ]]; then
    echo "ERROR: Not on main. Aborting."
    exit 1
fi

# ==========================================================
# 2. Back up manuscript files OUTSIDE Git repository
# ==========================================================

BACKUP="../MDSA-manuscript-backup-$(date +%Y%m%d_%H%M%S)"

mkdir -p "$BACKUP"

if [[ -d manuscript_data_availability ]]; then
    cp -a manuscript_data_availability "$BACKUP/"
fi

echo "Local file backup: $BACKUP"

# ==========================================================
# 3. Permanently ignore the directory
# ==========================================================

cat >> .gitignore <<'EOF'

# Manuscript analysis files - local only
/manuscript_data_availability/
EOF

# Commit all outstanding changes before rewriting history
git add -A

if ! git diff --cached --quiet; then
    git commit -m "Prepare repository cleanup"
fi

# Back up the ORIGINAL Git history
git bundle create "$BACKUP/pre-filter-history.bundle" --all

# ==========================================================
# 4. REMOVE ENTIRE DIRECTORY FROM ALL GIT HISTORY
# ==========================================================

python -m pip install git-filter-repo

git filter-repo --force \
    --invert-paths \
    --path manuscript_data_availability/

# ==========================================================
# 5. Restore local manuscript files (ignored by Git)
# ==========================================================

if [[ -d "$BACKUP/manuscript_data_availability" ]]; then
    mkdir -p manuscript_data_availability
    cp -a "$BACKUP/manuscript_data_availability/." \
        manuscript_data_availability/
fi

# ==========================================================
# 6. Verify cleanup
# ==========================================================

echo "Checking tracked files..."

if [[ -n "$(git ls-files -- manuscript_data_availability/)" ]]; then
    echo "ERROR: Manuscript directory still tracked!"
    exit 1
fi

echo "Checking historical files..."

if [[ -n "$(git log --all --format= --name-only -- manuscript_data_availability/)" ]]; then
    echo "ERROR: Manuscript directory still in Git history!"
    exit 1
fi

echo "SUCCESS: Directory removed from local Git history."

# ==========================================================
# 7. Re-add GitHub remote
# ==========================================================

git remote remove origin 2>/dev/null || true

git remote add origin \
    https://github.com/zeper-eng/MDSA-tools.git

git remote -v

# ==========================================================
# 8. FORCE PUSH REWRITTEN HISTORY
# ==========================================================

git push --force -u origin main

echo "DONE: GitHub main history rewritten successfully."