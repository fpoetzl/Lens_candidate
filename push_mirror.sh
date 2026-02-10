#!/bin/bash
set -e

MIRROR_REMOTE=mirror
MIRROR_BRANCH=main

# ----------------------------
# Commit message
# ----------------------------
# Use all arguments as commit message
COMMIT_MSG="$*"

# Default message if none provided
if [ -z "$COMMIT_MSG" ]; then
    COMMIT_MSG="Mirror push"
fi

# Files/dirs allowed in mirror
INCLUDE=(
  example.ipynb
  helpers.py
  __init__.py
  lens_candidate.py
  plotters.py
  README.md
)

git checkout main

# Create temporary orphan branch
git checkout --orphan mirror-tmp
git rm -rf .

# Bring in allowed files from main
git checkout main -- "${INCLUDE[@]}"

git add .
git commit -m "$COMMIT_MSG"

# Force push clean history to mirror
git push "$MIRROR_REMOTE" mirror-tmp:"$MIRROR_BRANCH" --force

# Return to main and delete temp branch
git checkout main
git branch -D mirror-tmp
