#!/usr/bin/env python3
"""
TataStrive Update Publisher
===========================
Run this script from your dev machine whenever you want to push an update
to all deployed clients.  It will:

  1. Bump the version string in app/__init__.py
  2. Build a manifest.json  —  SHA-256 hash of every tracked source file
  3. Fetch the previous manifest from the last GitHub Release to compute
     the delta (only changed / new files go into the patch)
  4. Build a patch.zip  —  contains only the changed files
  5. Create a GitHub Release (tag v{version}) and upload both assets

Clients running the app will detect the new release on next startup,
download only patch.zip (the delta), and restart automatically.

Usage
-----
    # Option A — pass token on the command line
    python scripts/publish_update.py --version 1.1.0 --token ghp_xxxx

    # Option B — set environment variables (recommended for CI)
    $env:GITHUB_TOKEN = "ghp_xxxx"
    $env:GITHUB_REPO  = "your-org/TataStriveFinal"
    python scripts/publish_update.py --version 1.1.0 --changelog "Fixed X, added Y"

    # Option C — dev machine: copy .env.publish.example to .env.publish and add PAT
    # (gitignored; never put PAT in .env or it may get bundled into the exe)

Prerequisites
-------------
    - Python 3.9+  (stdlib only — no third-party packages needed)
    - A GitHub Personal Access Token with  repo  scope
    - The repository must exist on GitHub and be configured in GITHUB_REPO below
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import zipfile
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional
from urllib.error import HTTPError
from urllib.request import Request, urlopen

REPO_ROOT_FOR_ENV = Path(__file__).resolve().parent.parent


def _load_dev_publish_env() -> None:
    """
    On your dev machine, put GITHUB_TOKEN (and optional GITHUB_REPO) in
    project-root `.env.publish` (copy from `.env.publish.example`).
    That file is gitignored and is never bundled by build_exe.py — unlike `.env`.
    """
    path = REPO_ROOT_FOR_ENV / ".env.publish"
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key == "GITHUB_TOKEN" and val and not os.environ.get("GITHUB_TOKEN"):
            os.environ["GITHUB_TOKEN"] = val
        if key == "GITHUB_REPO" and val and not os.environ.get("GITHUB_REPO"):
            os.environ["GITHUB_REPO"] = val


# ─────────────────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────────────────
GITHUB_API = "https://api.github.com"

REPO_ROOT = Path(__file__).resolve().parent.parent   # project root

# Glob patterns for files tracked by the updater
TRACKED_PATTERNS: List[str] = [
    "app/**/*.py",
    "classroom_analysis/**/*.py",
    "scripts/**/*.py",
    "run_app.py",
    "requirements_app.txt",
]

# Sub-strings that immediately disqualify a path from being tracked
EXCLUDE_SUBSTRINGS: List[str] = [
    "__pycache__",
    ".pyc",
    ".pyo",
    os.sep + "env" + os.sep,
    os.sep + "venv" + os.sep,
    os.sep + ".git" + os.sep,
    os.sep + "dist" + os.sep,
    os.sep + "build" + os.sep,
    # this script itself is excluded automatically (see collect_tracked_files)
]


# ─────────────────────────────────────────────────────────────────────────────
#  File utilities
# ─────────────────────────────────────────────────────────────────────────────

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65_536), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_tracked_files() -> Dict[str, str]:
    """Return {posix-relative-path: sha256} for every tracked file."""
    result: Dict[str, str] = {}
    this_script = Path(__file__).resolve()

    for pattern in TRACKED_PATTERNS:
        for p in REPO_ROOT.glob(pattern):
            if not p.is_file():
                continue
            if p.resolve() == this_script:
                continue
            p_str = str(p)
            if any(ex in p_str for ex in EXCLUDE_SUBSTRINGS):
                continue
            rel = p.relative_to(REPO_ROOT).as_posix()
            result[rel] = sha256(p)

    return result


def build_patch_zip(
    current_files: Dict[str, str],
    prev_files:    Dict[str, str],
) -> tuple[bytes, List[str]]:
    """
    Return (zip_bytes, list_of_changed_paths).
    A file is included if it is new or its hash differs from prev_files.
    If prev_files is empty every tracked file is included.
    """
    changed = {
        path: digest
        for path, digest in current_files.items()
        if prev_files.get(path) != digest
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for rel_path in changed:
            abs_path = REPO_ROOT / rel_path
            if abs_path.exists():
                zf.write(abs_path, rel_path)

    return buf.getvalue(), list(changed.keys())


# ─────────────────────────────────────────────────────────────────────────────
#  Version bumping
# ─────────────────────────────────────────────────────────────────────────────

def bump_version_in_source(version: str) -> None:
    """Update __version__ in app/__init__.py."""
    init_path = REPO_ROOT / "app" / "__init__.py"
    original  = init_path.read_text(encoding="utf-8")
    updated   = re.sub(
        r'(__version__\s*=\s*)["\'].*?["\']',
        rf'\g<1>"{version}"',
        original,
    )
    if updated == original:
        print(f"[Publisher] Version pattern not found in {init_path} — skipping bump.")
        return
    init_path.write_text(updated, encoding="utf-8")
    print(f"[Publisher] Bumped __version__ -> {version}")

    # Also patch app/main.py  (app.setApplicationVersion)
    main_path = REPO_ROOT / "app" / "main.py"
    if main_path.exists():
        mc = main_path.read_text(encoding="utf-8")
        mu = re.sub(
            r'(setApplicationVersion\s*\(\s*)["\'].*?["\']',
            rf'\g<1>"{version}"',
            mc,
        )
        if mu != mc:
            main_path.write_text(mu, encoding="utf-8")
            print(f"[Publisher] Patched setApplicationVersion -> {version}")


# ─────────────────────────────────────────────────────────────────────────────
#  GitHub API helpers  (stdlib urllib only)
# ─────────────────────────────────────────────────────────────────────────────

def _gh_request(
    method: str,
    url: str,
    token: str,
    body: Optional[bytes] = None,
    content_type: str = "application/json",
    timeout: int = 30,
) -> Dict:
    headers = {
        "Authorization": f"token {token}",
        "Accept":        "application/vnd.github+json",
        "User-Agent":    "TataStrivePublisher/1.0",
        "Content-Type":  content_type,
    }
    req = Request(url, data=body, headers=headers, method=method)
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        body_txt = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"GitHub API {exc.code} on {method} {url}:\n{body_txt}"
        ) from exc


def fetch_previous_manifest(repo: str, token: str) -> Optional[Dict]:
    """Return the manifest.json from the latest GitHub release, or None."""
    try:
        data = _gh_request("GET", f"{GITHUB_API}/repos/{repo}/releases/latest", token)
        for asset in data.get("assets", []):
            if asset["name"] == "manifest.json":
                req = Request(
                    asset["browser_download_url"],
                    headers={"User-Agent": "TataStrivePublisher/1.0"},
                )
                with urlopen(req, timeout=15) as resp:
                    return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        print(f"[Publisher] No previous manifest ({exc}) — treating as first release.")
    return None


def get_release_by_tag(repo: str, token: str, version: str) -> Optional[Dict]:
    tag = f"v{version.lstrip('v')}"
    try:
        return _gh_request(
            "GET",
            f"{GITHUB_API}/repos/{repo}/releases/tags/{tag}",
            token,
        )
    except RuntimeError as exc:
        if "404" in str(exc):
            return None
        raise


def create_github_release(repo: str, token: str, version: str, changelog: str) -> Dict:
    tag = f"v{version.lstrip('v')}"
    payload = json.dumps({
        "tag_name":         tag,
        "target_commitish": "main",
        "name":             tag,
        "body":             changelog or f"Release {tag}",
        "draft":            False,
        "prerelease":       False,
    }).encode("utf-8")
    return _gh_request("POST", f"{GITHUB_API}/repos/{repo}/releases", token, payload)


def get_or_create_release(repo: str, token: str, version: str, changelog: str) -> Dict:
    existing = get_release_by_tag(repo, token, version)
    if existing:
        print(f"[Publisher] Release {existing.get('html_url')} already exists — uploading assets.")
        return existing
    return create_github_release(repo, token, version, changelog)


def remove_release_assets(
    repo: str,
    token: str,
    release: Dict,
    names: set[str],
) -> None:
    """Delete existing release assets so re-upload does not 422 (duplicate name)."""
    for asset in release.get("assets", []):
        name = asset.get("name")
        if name not in names:
            continue
        asset_id = asset.get("id")
        if not asset_id:
            continue
        _gh_request(
            "DELETE",
            f"{GITHUB_API}/repos/{repo}/releases/assets/{asset_id}",
            token,
        )
        print(f"[Publisher] Removed existing asset: {name}")


def upload_release_asset(
    upload_url: str,
    token:      str,
    name:       str,
    data:       bytes,
    mime:       str,
) -> None:
    url = upload_url.split("{")[0] + f"?name={name}"
    headers = {
        "Authorization": f"token {token}",
        "Content-Type":  mime,
        "User-Agent":    "TataStrivePublisher/1.0",
    }
    req = Request(url, data=data, headers=headers, method="POST")
    with urlopen(req, timeout=120) as resp:
        resp.read()
    size_kb = len(data) / 1024
    print(f"[Publisher] Uploaded  {name}  ({size_kb:,.0f} KB)")


# ─────────────────────────────────────────────────────────────────────────────
#  Main publish workflow
# ─────────────────────────────────────────────────────────────────────────────

def publish(
    version: str,
    token: str,
    repo: str,
    changelog: str,
    *,
    no_bump: bool = False,
) -> None:
    print(f"\n{'='*60}")
    print(f"  TataStrive Publisher  --  v{version}  ->  {repo}")
    print(f"{'='*60}\n")

    # 1. Bump version in source files
    if not no_bump:
        bump_version_in_source(version)
    else:
        print("[Publisher] --no-bump: keeping __version__ in source unchanged.")

    # 2. Collect current tracked files
    current_files = collect_tracked_files()
    print(f"[Publisher] Tracking {len(current_files)} file(s)")

    # 3. Fetch previous manifest → compute delta
    prev_manifest = fetch_previous_manifest(repo, token)
    prev_files: Dict[str, str] = {}
    if prev_manifest:
        prev_files = {f["path"]: f["sha256"] for f in prev_manifest.get("files", [])}
        print(f"[Publisher] Previous manifest has {len(prev_files)} file(s)")
    else:
        print("[Publisher] No previous release — patch will include ALL tracked files")

    # 4. Build patch.zip
    patch_bytes, changed = build_patch_zip(current_files, prev_files)
    print(f"[Publisher] patch.zip  —  {len(changed)} file(s), "
          f"{len(patch_bytes)/1024:,.0f} KB")
    for p in changed:
        print(f"           + {p}")

    if not changed:
        print("\n[Publisher] Nothing changed since last release — aborting.")
        sys.exit(0)

    # 5. Build manifest.json  (full manifest of all tracked files)
    manifest = {
        "version":      version,
        "release_date": date.today().isoformat(),
        "changelog":    changelog or f"Release v{version}",
        "files": [
            {
                "path":   p,
                "sha256": h,
                "size":   (REPO_ROOT / p).stat().st_size,
            }
            for p, h in current_files.items()
            if (REPO_ROOT / p).exists()
        ],
    }
    manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
    print(f"[Publisher] manifest.json  —  {len(manifest['files'])} entries, "
          f"{len(manifest_bytes)/1024:.1f} KB")

    # 6. Create GitHub Release (or attach assets to tag that already exists)
    print("\n[Publisher] Creating / resolving GitHub release…")
    release    = get_or_create_release(repo, token, version, changelog)
    upload_url = release["upload_url"]
    print(f"[Publisher] Release URL: {release['html_url']}")

    # 7. Upload assets (replace any existing manifest.json / patch.zip first)
    asset_names = {"manifest.json", "patch.zip"}
    remove_release_assets(repo, token, release, asset_names)
    upload_release_asset(upload_url, token, "manifest.json", manifest_bytes, "application/json")
    upload_release_asset(upload_url, token, "patch.zip",     patch_bytes,    "application/zip")

    print(f"\n[OK] Release v{version} is live.")
    print(f"   {release['html_url']}")
    print(f"   Clients will auto-update on next startup.\n")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    _load_dev_publish_env()
    repo_default = os.getenv("GITHUB_REPO", "Pranav-stac/TataStriveFinal")

    parser = argparse.ArgumentParser(
        description="Publish a TataStrive delta update to GitHub Releases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--version", "-v", required=True,
        help="New semantic version, e.g. 1.1.0",
    )
    parser.add_argument(
        "--token", "-t",
        default=os.getenv("GITHUB_TOKEN"),
        help="GitHub Personal Access Token (or set $GITHUB_TOKEN, or .env.publish)",
    )
    parser.add_argument(
        "--repo", "-r",
        default=repo_default,
        help='GitHub repo in owner/name format (or set $GITHUB_REPO, or .env.publish)',
    )
    parser.add_argument(
        "--changelog", "-c",
        default="",
        help="Release notes / changelog text (optional)",
    )
    parser.add_argument(
        "--no-bump",
        action="store_true",
        help="Do not modify app/__init__.py (use when version already bumped in git)",
    )
    args = parser.parse_args()

    if not args.token:
        parser.error(
            "No GitHub token provided.\n"
            "Pass --token ghp_xxxx  or  set the GITHUB_TOKEN environment variable."
        )

    if args.repo in ("OWNER/TataStriveFinal", ""):
        parser.error(
            "Please set --repo owner/repo  or the GITHUB_REPO env var "
            "to your actual GitHub repository."
        )

    publish(
        args.version,
        args.token,
        args.repo,
        args.changelog,
        no_bump=args.no_bump,
    )


if __name__ == "__main__":
    main()
