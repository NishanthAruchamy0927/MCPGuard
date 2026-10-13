"""Dataset acquisition.

Sources
-------
1. **BFCL v3** (Berkeley Function-Calling Leaderboard, Hugging Face dataset
   ``gorilla-llm/Berkeley-Function-Calling-Leaderboard``, Apache-2.0): real
   function/tool definitions (name, description, JSON-schema parameters), the
   real user questions that target them and the gold ("possible answer") calls.
   BFCL functions use the same JSON-schema contract as MCP ``inputSchema`` and
   are converted to MCP tool format during dataset construction.
2. **MCP reference servers**: curated tool definitions (see
   :mod:`src.data.mcp_reference`).

Usage::

    python -m src.data.download            # download if missing
    python -m src.data.download --force    # re-download
"""
from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import requests

from src.data import mcp_reference
from src.utils.config import load_config
from src.utils.io import write_json
from src.utils.logging_utils import get_logger, log_event

log = get_logger("data.download")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(url: str, dest: Path, force: bool) -> Path:
    if dest.exists() and not force and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def download(force: bool = False) -> dict:
    cfg = load_config()
    bcfg = cfg["data"]["bfcl"]
    raw_dir = cfg.path("raw") / "bfcl"
    manifest = {"retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "license": "Apache-2.0 (BFCL)", "files": []}
    for cat in bcfg["categories"]:
        fname = f"{bcfg['version_prefix']}{cat}.json"
        for kind, url, dest in [
            ("functions", f"{bcfg['base_url']}/{fname}", raw_dir / fname),
            ("possible_answer", f"{bcfg['base_url']}/possible_answer/{fname}", raw_dir / f"pa_{fname}"),
        ]:
            try:
                _fetch(url, dest, force)
                manifest["files"].append({"category": cat, "kind": kind, "url": url, "path": str(dest.relative_to(cfg.path("raw").parent.parent)),
                                          "bytes": dest.stat().st_size, "sha256": _sha256(dest)})
                log_event(log, "downloaded", category=cat, kind=kind, bytes=dest.stat().st_size)
            except requests.RequestException as exc:
                log_event(log, "download failed", category=cat, kind=kind, error=str(exc))
    tools = mcp_reference.build()
    from src.utils.config import resolve
    write_json(tools, resolve(cfg["data"]["mcp_reference_file"]))
    manifest["mcp_reference_tools"] = len(tools)
    write_json(manifest, cfg.path("metadata") / "download_manifest.json")
    log_event(log, "acquisition complete", bfcl_files=len(manifest["files"]), mcp_reference_tools=len(tools))
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true")
    download(force=ap.parse_args().force)


if __name__ == "__main__":
    main()
