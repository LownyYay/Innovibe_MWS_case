"""Resolve a v2.1 Kaggle input directory or extract official runtime files."""
import hashlib
from pathlib import Path
from zipfile import ZipFile

from data_v21 import CONFIG, REQUIRED, digest, verify_data


def resolve_data(input_root, destination, data_path="", archive_path=""):
    if data_path:
        verify_data(data_path)
        return Path(data_path)
    root = Path(input_root)
    if not archive_path:
        directories = [p.parent for p in root.rglob("train.jsonl")
                       if digest(p) == CONFIG["data_sha256"]["train.jsonl"]]
        for candidate in sorted(set(directories)):
            if REQUIRED <= {p.name for p in candidate.iterdir()}:
                verify_data(candidate)
                return candidate
    candidates = [Path(archive_path)] if archive_path else sorted(root.rglob("*.zip"))
    for archive in candidates:
        with ZipFile(archive) as package:
            members = {}
            for name in package.namelist():
                basename = Path(name).name
                if basename in REQUIRED:
                    if basename in members:
                        raise ValueError(f"Duplicate runtime member: {basename}")
                    members[basename] = name
            if set(members) != REQUIRED:
                continue
            contents = {basename: package.read(name) for basename, name in members.items()}
            if hashlib.sha256(contents["train.jsonl"]).hexdigest() != CONFIG["data_sha256"]["train.jsonl"]:
                continue
            dest = Path(destination)
            dest.mkdir(parents=True, exist_ok=True)
            for name, value in contents.items():
                (dest / name).write_bytes(value)
            verify_data(dest)
            return dest
    raise FileNotFoundError("Attach v2.1 data or set DATA_PATH / ARCHIVE_PATH")
