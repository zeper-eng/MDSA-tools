"""Download selected, peer-reviewed SETD8 apo-SETD8 WT and A296T trajectories.

SETD8 OSF wiki (https://osf.io/2h6p4/wiki):
  APO/11707/run0-cloneN.h5       = WT from 1ZKK chain A
  MUTANTS/11713/run9-cloneN.h5   = A296T from 1ZKK chain A

Usage:
  python grab_osf_trajectories_fixed.py --dry-run --out "C:/Users/Luis/Projects/MDSA_tools_revisions/Trajectories_and_topologies/SETD8"
  python grab_osf_trajectories_fixed.py --out "C:/Users/Luis/Projects/MDSA_tools_revisions/Trajectories_and_topologies/SETD8"

Optional for larger OSF rate limits: set OSF_TOKEN in your shell first.
"""
import argparse
import os
import re
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

import requests

NODE_ID = "2h6p4"
TARGETS = {
    "WT": ("APO", "11707", "run0-clone"),
    "A296T": ("MUTANTS", "11713", "run9-clone"),
}


def with_params(url, **params):
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query))
    query.update(params)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def get(session, url, **kwargs):
    try:
        response = session.get(url, timeout=(15, 90), **kwargs)
    except requests.RequestException as exc:
        raise RuntimeError(f"Network error at {url}: {exc}") from exc
    if response.status_code == 429:
        retry_after = response.headers.get("Retry-After")
        response.close()
        extra = f" Retry-After: {retry_after}." if retry_after else ""
        raise RuntimeError(f"OSF API returned 429 (rate limit).{extra} Don't repeatedly retry; use OSF_TOKEN or let the quota reset.")
    response.raise_for_status()
    return response


def list_items(session, url):
    # Large page size substantially reduces API calls. The server may cap this.
    url = with_params(url, **{"page[size]": "100"})
    while url:
        response = get(session, url)
        payload = response.json()
        yield from payload.get("data", [])
        next_url = payload.get("links", {}).get("next")
        url = next_url.get("href") if isinstance(next_url, dict) else next_url


def children_url(item):
    rel = item.get("relationships", {}).get("files", {}).get("links", {}).get("related")
    if isinstance(rel, dict):
        return rel.get("href")
    return rel


def find_folder(session, parent_url, name):
    for item in list_items(session, parent_url):
        if item.get("attributes", {}).get("kind") == "folder" and item.get("attributes", {}).get("name", "").casefold() == name.casefold():
            url = children_url(item)
            if not url:
                raise RuntimeError(f"Folder {name} has no OSF file-list URL")
            return url
    raise RuntimeError(f"Folder {name} not found at {parent_url}")


def discover(session, label, top_folder, subfolder, file_prefix):
    root = f"https://api.osf.io/v2/nodes/{NODE_ID}/files/osfstorage/"
    parent = find_folder(session, root, top_folder)
    folder = find_folder(session, parent, subfolder)
    # Filter is supported by OSF's file-list API. Still check names explicitly.
    filtered = with_params(folder, **{"filter[name]": file_prefix})
    matches = {}
    pat = re.compile(rf"^{re.escape(file_prefix)}(\d+)\.h5$", re.IGNORECASE)
    for item in list_items(session, filtered):
        attrs = item.get("attributes", {})
        name = attrs.get("name", "")
        m = pat.fullmatch(name)
        if not m or attrs.get("kind") != "file":
            continue
        download_url = item.get("links", {}).get("download")
        if download_url:
            matches[int(m.group(1))] = (name, download_url, attrs.get("size"))
    return matches


def download(session, url, dest, size):
    if dest.exists() and (size is None or dest.stat().st_size == size):
        print("SKIP", dest, "(already present)", flush=True)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    with get(session, url, stream=True) as response:
        with partial.open("wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    handle.write(chunk)
    if size is not None and partial.stat().st_size != size:
        raise RuntimeError(f"Incomplete download {dest.name}: expected {size}, got {partial.stat().st_size}")
    partial.replace(dest)
    print("SAVED", dest, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("SETD8_trajectories"))
    ap.add_argument("--count", type=int, default=30)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.count < 1:
        ap.error("--count must be >=1")
    session = requests.Session()
    session.headers["User-Agent"] = "SETD8-JCIM-trajectories/1.1"
    token = os.environ.get("OSF_TOKEN")
    if token:
        session.headers["Authorization"] = f"Bearer {token}"
        print("OSF_TOKEN detected: using authenticated API access", flush=True)
    else:
        print("Unauthenticated OSF API; likely subject to 100 requests/hour quota", flush=True)
    chosen = {}
    for label, (top, sub, prefix) in TARGETS.items():
        print(f"Discovering {label}: {top}/{sub}/{prefix}*.h5", flush=True)
        matches = discover(session, label, top, sub, prefix)
        ids = sorted(matches)[:args.count]
        if len(ids) < args.count:
            raise RuntimeError(f"Found {len(ids)} {label} trajectories, need {args.count}. No downloads started.")
        chosen[label] = [matches[i] for i in ids]
        print(f"{label}: found {len(matches)} trajectories; selected clone IDs {ids}", flush=True)
    if args.dry_run:
        print("Dry run complete; no downloads started.")
        return
    for label, entries in chosen.items():
        for name, url, size in entries:
            download(session, url, args.out / label / name, size)
    print("DONE:", args.out.resolve())


if __name__ == "__main__":
    main()
