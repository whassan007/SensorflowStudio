#!/usr/bin/env python
"""Deploy Sensorflow Studio to a Hugging Face Space (Docker SDK).

Idempotent / re-runnable: creates the Space if missing, uploads the current
repo state (frontend is built inside the Docker image on HF's side), waits for
the build, and smoke-tests the deployed endpoints.

Usage:
    .venv/bin/python deploy/huggingface/deploy.py [--space-name NAME] [--no-wait]

Requires a Hugging Face token with write access (`hf auth login` or HF_TOKEN).
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import LocalTokenNotFoundError

REPO_ROOT = Path(__file__).resolve().parents[2]

# Mirrors .dockerignore, plus files that must not reach the Space repo as-is.
IGNORE_PATTERNS = [
    "node_modules/**",
    ".venv/**",
    ".git/**",
    "runs/**",
    "dist/**",
    "models/**",
    "*.pt",
    "**/*.pt",
    "**/__pycache__/**",
    ".pytest_cache/**",
    ".claude/**",
    ".DS_Store",
    "**/.DS_Store",
    "README.md",  # root project README; the Space README is uploaded separately
]

TERMINAL_ERROR_STAGES = {"BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR", "DELETING"}


def http_check(url: str, method: str = "GET", body: dict | None = None) -> str:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json"} if body is not None else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            snippet = resp.read(200).decode(errors="replace").replace("\n", " ")
            return f"{resp.status} {snippet[:120]}"
    except Exception as exc:  # noqa: BLE001 - smoke test, report anything
        return f"FAILED: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space-name", default="sensorflow-studio")
    parser.add_argument("--no-wait", action="store_true",
                        help="upload only; don't wait for the build")
    parser.add_argument("--timeout-minutes", type=float, default=15)
    args = parser.parse_args()

    api = HfApi()
    try:
        user = api.whoami()["name"]
    except LocalTokenNotFoundError:
        print("ERROR: no Hugging Face token found. Run `hf auth login` "
              "(or set HF_TOKEN) with a WRITE token, then re-run this script.")
        return 1

    repo_id = f"{user}/{args.space_name}"
    print(f"Deploying to Space: {repo_id}")

    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)

    print("Uploading repository files...")
    api.upload_folder(
        folder_path=str(REPO_ROOT),
        repo_id=repo_id,
        repo_type="space",
        ignore_patterns=IGNORE_PATTERNS,
        commit_message="Deploy Sensorflow Studio",
    )
    # The Space README (with docker frontmatter) replaces the project README.
    api.upload_file(
        path_or_fileobj=str(REPO_ROOT / "deploy" / "huggingface" / "README.md"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="space",
        commit_message="Update Space README",
    )

    space_url = f"https://{user}-{args.space_name.replace('_', '-')}.hf.space"
    print(f"Space page: https://huggingface.co/spaces/{repo_id}")
    print(f"App URL:    {space_url}/")

    if args.no_wait:
        return 0

    print("Waiting for the Space build (this can take several minutes)...")
    deadline = time.time() + args.timeout_minutes * 60
    stage = None
    while time.time() < deadline:
        runtime = api.get_space_runtime(repo_id)
        if runtime.stage != stage:
            stage = runtime.stage
            print(f"  stage: {stage}")
        if stage == "RUNNING":
            break
        if stage in TERMINAL_ERROR_STAGES:
            print(f"Build/runtime failed (stage={stage}). Check logs at "
                  f"https://huggingface.co/spaces/{repo_id}?logs=build")
            return 2
        time.sleep(15)
    else:
        print(f"Timed out after {args.timeout_minutes} min (last stage: {stage}).")
        return 3

    print("Space is RUNNING. Verifying endpoints...")
    print(f"  GET  /            -> {http_check(space_url + '/')}")
    print(f"  GET  /dashboard/  -> {http_check(space_url + '/dashboard/')}")
    print(f"  POST /api/ssam/statewide -> "
          f"{http_check(space_url + '/api/ssam/statewide', 'POST', {'page': 1, 'page_size': 1})}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
