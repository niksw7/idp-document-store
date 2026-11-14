#!/usr/bin/env python3
"""run_analysis.py — the sandbox bootstrap for the IDP Claw agent.

This is the ONLY script the agent is instructed to run. It is fetched
from the pinned URL in the agent's instructions and verified against the
SHA-256 in those instructions; then it:

  1. fetches the analyzer + rules file from the SAME pinned repo,
     verifying each against the SHA-256 pins below (a mismatch is a
     tamper/error signal: refuse loudly, fetch nothing else),
  2. installs the three runtime dependencies (pypdfium2, pillow, pyyaml)
     if missing, from the policy-allowed package indexes,
  3. executes the analyzer with the caller's arguments and passes its
     exit code through.

Network discipline: this script touches ONLY raw.githubusercontent.com
(the same domain the capability policy allows for document fetches).
Repo, ref and paths are constants. The document fetch itself is inside
the analyzer, pinned the same way.
"""

from __future__ import annotations

import hashlib
import importlib.util
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

RAW_BASE = (
    "https://raw.githubusercontent.com/niksw7/idp-document-store/main/agent"
)
REQUEST_TIMEOUT_SECONDS = 60

#: sha256 of the exact bytes raw.githubusercontent.com serves — refreshed
#: by agentworks/scripts/build_claw_spec.py at release time
PINNED_FILES: dict[str, str] = {
    "idp_analyze.py": "90ff0b9c52fc5482faa5f0d3ae7371fe02f7439861ce6038c13ce8cb87118b0b",
    "document_rules.yaml": "633d4842d50d2e4bc3661ed7d8b9b5811be447b6404e834de842cee1f2ccfdf8",
}

#: runtime dependencies (installed only from the policy-allowed indexes)
DEPENDENCIES = ["pypdfium2", "pillow", "pyyaml"]

#: the analyzer entrypoint, staged next to this file
ANALYZER_NAME = "idp_analyze.py"


def _fetch(name: str) -> bytes:
    url = f"{RAW_BASE}/{name}"
    request = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        raise SystemExit(
            f"error: could not fetch {name} (HTTP {exc.code}) from {url}"
        ) from exc
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"error: could not reach the code host for {name}: {exc.reason}"
        ) from exc


def _ensure_pinned_file(name: str, expected: str, target: Path) -> None:
    """Serve the pinned file from disk when its hash matches; else fetch,
    verify, write. A hash mismatch is fatal — never run unverified code."""
    if target.is_file():
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if digest == expected:
            return
        target.unlink()  # stale/tampered copy: re-fetch
    data = _fetch(name)
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected:
        raise SystemExit(
            f"error: {name} failed verification (sha256 {digest} != pinned "
            f"{expected}). Refusing to run unverified code — check that the "
            "agent instructions carry the current hash."
        )
    target.write_bytes(data)
    print(f"[bootstrap] verified + staged {name} (sha256 {digest[:12]}…)")


def _ensure_dependencies() -> None:
    missing = []
    for module, dist in (("pypdfium2", "pypdfium2"), ("PIL", "pillow"), ("yaml", "pyyaml")):
        try:
            __import__(module)
        except ImportError:
            missing.append(dist)
    if not missing:
        return
    if shutil.which("pip") is None and shutil.which("pip3") is None:
        raise SystemExit(
            "error: dependencies missing and pip unavailable: " + ", ".join(missing)
        )
    print(f"[bootstrap] installing: {', '.join(missing)}")
    pip = shutil.which("pip") or shutil.which("pip3")
    result = subprocess.run(
        [pip, "install", "--quiet", *missing],
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit("error: dependency installation failed")


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve().parent
    for name, expected in PINNED_FILES.items():
        _ensure_pinned_file(name, expected, here / name)
    _ensure_dependencies()

    spec = importlib.util.spec_from_file_location("idp_analyze", here / ANALYZER_NAME)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main(argv if argv is not None else sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
