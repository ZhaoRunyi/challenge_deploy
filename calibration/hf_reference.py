"""Download selected public dataset files with pinned revision and content hashes."""
import hashlib
from pathlib import Path
import time

import requests
from huggingface_hub import HfApi, hf_hub_url


def checked_file(path, metadata):
    if not path.is_file() or path.stat().st_size != metadata.size:
        return False
    digest = hashlib.sha256() if metadata.lfs else hashlib.sha1()
    if not metadata.lfs:
        digest.update(f"blob {metadata.size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest() == (metadata.lfs.sha256 if metadata.lfs else metadata.blob_id)


def download(repo, revision, paths, cache):
    api = HfApi(token=False)
    metadata = {entry.path: entry for entry in api.get_paths_info(repo, paths, repo_type="dataset", revision=revision)}
    for name in paths:
        path = Path(cache) / name
        if not path.resolve().is_relative_to(Path(cache).resolve()):
            raise ValueError("Invalid dataset path")
        entry = metadata[name]
        if checked_file(path, entry):
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".part")
        for attempt in range(3):
            try:
                # A plain GET avoids reliance on proxy-stripped HF HEAD headers.
                with requests.get(hf_hub_url(repo, name, repo_type="dataset", revision=revision), stream=True, timeout=(15, 45)) as response:
                    response.raise_for_status()
                    total = 0
                    with temporary.open("wb") as stream:
                        for block in response.iter_content(1024 * 1024):
                            total += len(block)
                            if total > entry.size:
                                raise ValueError(f"Unexpected download size: {name}")
                            stream.write(block)
                if not checked_file(temporary, entry):
                    raise ValueError(f"Dataset checksum mismatch: {name}")
                temporary.replace(path)
                break
            except (requests.RequestException, ValueError):
                temporary.unlink(missing_ok=True)
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
    return Path(cache)
