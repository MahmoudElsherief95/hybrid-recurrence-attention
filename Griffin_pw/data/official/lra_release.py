from __future__ import annotations

import hashlib
import os
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


LRA_RELEASE_URL = "https://storage.googleapis.com/long-range-arena/lra_release.gz"
LRA_RELEASE_ENV_ARCHIVE_PATH = "LRA_RELEASE_ARCHIVE_PATH"


DEFAULT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123 Safari/537.36"


@dataclass(frozen=True)
class LRAReleasePaths:
    archive_path: Path
    extract_dir: Path
    marker_path: Path


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_extract_tar(tar: tarfile.TarFile, dest: Path) -> None:
    dest = dest.resolve()
    for member in tar.getmembers():
        member_path = (dest / member.name).resolve()
        if not str(member_path).startswith(str(dest) + os.sep):
            raise RuntimeError(f"Unsafe path in tar archive: {member.name}")
    tar.extractall(dest)


def get_default_lra_cache_dir(repo_root: Path) -> Path:
    return repo_root / "Griffin_pw" / "data_cache" / "lra_release"


def ensure_lra_release_extracted(
    cache_dir: Path,
    *,
    url: str = LRA_RELEASE_URL,
    expected_sha256: Optional[str] = None,
    force_redownload: bool = False,
    user_agent: str = DEFAULT_USER_AGENT,
) -> LRAReleasePaths:
    """Ensure the LRA release archive is downloaded+extracted.

    This uses the official LRA public release archive from Google Cloud Storage.

    Returns paths to the archive and extracted directory.
    """

    cache_dir = cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    archive_path = cache_dir / "lra_release.gz"
    extract_dir = cache_dir / "extracted"
    marker_path = extract_dir / ".extracted_ok"

    paths = LRAReleasePaths(archive_path=archive_path, extract_dir=extract_dir, marker_path=marker_path)

    if marker_path.exists() and extract_dir.exists() and not force_redownload:
        return paths

    env_archive = os.environ.get(LRA_RELEASE_ENV_ARCHIVE_PATH)
    if env_archive:
        env_path = Path(env_archive).expanduser().resolve()
        if not env_path.exists():
            raise RuntimeError(f"{LRA_RELEASE_ENV_ARCHIVE_PATH} is set but file does not exist: {env_path}")
        archive_path = env_path
        paths = LRAReleasePaths(archive_path=archive_path, extract_dir=extract_dir, marker_path=marker_path)
        force_redownload = False

    if force_redownload and archive_path.exists() and archive_path.parent == cache_dir:
        archive_path.unlink()

    if not archive_path.exists():
        req = urllib.request.Request(url, headers={"User-Agent": user_agent})
        try:
            with urllib.request.urlopen(req) as resp:
                tmp_path = archive_path.with_suffix(".gz.part")
                with tmp_path.open("wb") as f:
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                tmp_path.replace(archive_path)
        except Exception as e:
            raise RuntimeError(
                "Failed to download the official LRA release archive. "
                "If your network blocks this URL, download it manually and set "
                f"{LRA_RELEASE_ENV_ARCHIVE_PATH} to the local file path. "
                f"URL={url}"
            ) from e

    if expected_sha256 is not None:
        got = _sha256_file(archive_path)
        if got.lower() != expected_sha256.lower():
            raise RuntimeError(
                "LRA release archive sha256 mismatch. "
                f"expected={expected_sha256} got={got}"
            )

    if extract_dir.exists():
        # best-effort clean if a previous extraction failed
        for p in sorted(extract_dir.glob("**/*"), reverse=True):
            try:
                if p.is_file() or p.is_symlink():
                    p.unlink()
                elif p.is_dir():
                    p.rmdir()
            except OSError:
                pass

    extract_dir.mkdir(parents=True, exist_ok=True)

    # The upstream file is named *.gz but is a tarball in practice.
    try:
        with tarfile.open(archive_path, mode="r:gz") as tar:
            _safe_extract_tar(tar, extract_dir)
    except tarfile.ReadError as e:
        raise RuntimeError(
            f"Failed to read/extract LRA release archive: {archive_path}. "
            "If the file is not a tar.gz, download it manually and point to your dataset directory."
        ) from e

    marker_path.write_text("ok\n", encoding="utf-8")
    return paths
