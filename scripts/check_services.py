#!/usr/bin/env python3
"""
Health check utility for SCYLLA's two backend services.

Pings the C++ Crow service (:8080) and the Python FastAPI service (:6900),
printing one line per service with HTTP status and raw JSON body, or
"UNREACHABLE" plus the exception detail if the call fails.

Usage:
    python scripts/check_services.py

Exit codes:
    0 - both services reachable
    1 - one or both services unreachable

Dependencies:
    Standard library only (urllib, json). No third-party imports.
"""

import json
import sys
import urllib.request

SERVICES = [
    ("C++ Crow (:8080)", "http://127.0.0.1:8080/health"),
    ("Python FastAPI (:6900)", "http://127.0.0.1:6900/health"),
]

TIMEOUT_SECONDS = 3


def check_service(name, url):
    """Return (status_line) describing reachability of a single service."""
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
            status = resp.status
            raw = resp.read().decode("utf-8", errors="replace")
            return f"{name}: HTTP {status} -> {raw}"
    except Exception as exc:  # noqa: BLE001 - report any failure as UNREACHABLE
        return f"{name}: UNREACHABLE -> {type(exc).__name__}: {exc}"


def main():
    all_ok = True
    for name, url in SERVICES:
        line = check_service(name, url)
        print(line)
        if "UNREACHABLE" in line:
            all_ok = False
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
