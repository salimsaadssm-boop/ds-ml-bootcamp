
import os
import sys
import subprocess
from pathlib import Path

PR_AUTHOR = os.environ.get("PR_AUTHOR") or ""
BASE_SHA = os.environ.get("BASE_SHA") or ""
HEAD_SHA = os.environ.get("HEAD_SHA") or ""

ALLOWED_ROOT = Path("submissions") / PR_AUTHOR  # submissions/<username>/
MAX_FILE_SIZE_MB = 50  # reject huge files
REQUIRED_ANY = {".ipynb", ".md", ".csv", ".pdf", ".doc", ".docx"}  # acceptable formats

def fail(msg):
    print(f"❌ {msg}")
    sys.exit(1)

def warn(msg):
    print(f"⚠️  {msg}")

def ok(msg):
    print(f"✅ {msg}")

if not PR_AUTHOR:
    fail("Could not detect PR author.")

# Get changed files between base and head
try:
    diff_files = subprocess.check_output(
        ["git", "diff", "--name-only", f"{BASE_SHA}", f"{HEAD_SHA}"], text=True
    ).strip().splitlines()
except subprocess.CalledProcessError as e:
    fail(f"Failed to compute diff: {e}")

if not diff_files:
    fail("No files changed in this PR.")

ok(f"PR author detected: {PR_AUTHOR}")
ok(f"Changed files count: {len(diff_files)}")

# 1) Ensure every changed file lives under submissions/<PR_AUTHOR>/
for f in diff_files:
    p = Path(f)
    if not (str(p).startswith(f"submissions/{PR_AUTHOR}/")):
        fail(f"File '{f}' is outside your folder 'submissions/{PR_AUTHOR}/'. Only submit within your own folder.")

# 2) Enforce case-sensitive username folder
for f in diff_files:
    parts = Path(f).parts
    if len(parts) < 2:
        fail(f"Unexpected path '{f}'. Must be within 'submissions/{PR_AUTHOR}/'.")
    if parts[0] != "submissions":
        fail(f"Unexpected top-level folder '{parts[0]}' in '{f}'.")
    if parts[1] != PR_AUTHOR:
        fail(f"Username folder must exactly match GitHub login (case-sensitive). Found '{parts[1]}', expected '{PR_AUTHOR}'.")

# 3) Check for disallowed file types (executables/binaries)
DISALLOWED_EXTS = {".exe", ".dll", ".bin", ".iso"}
for f in diff_files:
    if Path(f).suffix.lower() in DISALLOWED_EXTS:
        fail(f"Disallowed file type detected: '{f}'. Remove binaries/executables.")

# 4) Enforce file size limits
for f in diff_files:
    if not Path(f).exists():
        continue
    size_mb = Path(f).stat().st_size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        fail(f"'{f}' is too large ({size_mb:.1f} MB). Max allowed is {MAX_FILE_SIZE_MB} MB.")

# 5) Require at least one acceptable file format
has_required = any(Path(f).suffix.lower() in REQUIRED_ANY for f in diff_files)
if not has_required:
    fail("Submission must include at least one of: .ipynb, .md, .csv, .pdf, .doc, or .docx file.")

# 6) Optional: Suggest structure if files are dumped at root
for f in diff_files:
    parts = Path(f).parts
    if len(parts) == 2:  # exactly submissions/<user>
        warn("Consider organizing your work into assignment subfolders (e.g., assignment1/, assignment2/).")

ok("All pre-checks passed. Maintainers can now review your work. 🚀")
