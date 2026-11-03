import os
import hashlib
import subprocess
from datetime import datetime


def compute_sha256(filepath):
    """Compute SHA-256 hash of a file for experiment reproducibility and provenance."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def get_git_commit(cwd=None):
    """Retrieve current Git commit hash if in a Git repository."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd or os.getcwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return res.stdout.strip() if res.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def log_model_manifest(manifest_csv_path, run_id, model_path, git_commit=None):
    """
    Append an entry to models_manifest.csv (tracked in Git per Section 9.2).
    Format: run_id \t local_path \t sha256 \t git_commit \t date
    """
    if not os.path.exists(model_path):
        return

    sha256_val = compute_sha256(model_path)
    commit = git_commit or get_git_commit(os.path.dirname(manifest_csv_path))
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    file_exists = os.path.exists(manifest_csv_path)
    with open(manifest_csv_path, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write("run_id\tlocal_path\tsha256\tgit_commit\tdate\n")
        f.write(f"{run_id}\t{model_path}\t{sha256_val}\t{commit}\t{date_str}\n")
