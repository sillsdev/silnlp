#!/usr/bin/env python3
"""Diagnose a local SILNLP setup.

Checks environment variables, connectivity to the MinIO bucket (usually over a VPN),
the rclone configuration and mount, and ClearML credentials, then prints a report
with a remediation hint for every failure.

The script uses only the Python standard library so that it runs even when the
poetry/conda environment is broken or missing:

    python silnlp/common/check_setup.py

Exit code is 0 if no check fails (warnings are allowed), 1 otherwise.
"""

import argparse
import base64
import configparser
import os
import platform
import re
import shutil
import socket
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple
from urllib.parse import urlparse

OK = "OK"
FAIL = "FAIL"
WARN = "WARN"
SKIP = "SKIP"
ICONS = {OK: "✅", FAIL: "❌", WARN: "⚠️ ", SKIP: "⏭️ "}

TIMEOUT = 8
DEFAULT_CLEARML_API_HOST = "https://api.sil.hosted.allegro.ai"

# On WSL the VPN runs on the Windows host, so VPN hints must point there.
IS_WSL = "microsoft" in platform.uname().release.lower()

if IS_WSL:
    VPN_DOWN_HINT = (
        "You are on WSL, so the VPN must be running on Windows (outside WSL) — check that the "
        "tunnel is active in the Windows WireGuard app. WSL may also not use the VPN's DNS server; "
        "if the VPN is up but resolution still fails, pin the address: set MINIO_ENDPOINT_IP in "
        "your env vars file and run 'source ./setup_env_vars.sh <file>' to write it to /etc/hosts."
    )
    VPN_ROUTE_HINT = (
        "DNS works but traffic to {ip} is not getting through. On WSL the VPN runs on Windows — "
        "check in the Windows WireGuard app that the tunnel is active and that its AllowedIPs "
        "cover {ip}. If they do not, ask the operations team for an updated VPN config or for a "
        "MINIO_ENDPOINT_IP inside the routed range to pin in /etc/hosts. Note that 'ping' is not "
        "a reliable test — ICMP may be blocked even when the endpoint works."
    )
else:
    VPN_DOWN_HINT = (
        "The VPN is probably down or its DNS server is not being used. Check the tunnel "
        "(e.g. 'ip a show wg0'), or pin the address: set MINIO_ENDPOINT_IP in your env vars "
        "file and run 'source ./setup_env_vars.sh <file>' to write it to /etc/hosts."
    )
    VPN_ROUTE_HINT = (
        "DNS works but traffic to {ip} is not getting through, which usually means the VPN "
        "does not route that address. Check that the resolved IP is covered by the WireGuard "
        "AllowedIPs ('sudo wg show wg0 allowed-ips') and the routing table ('ip route'). If it "
        "is not, ask the operations team for an updated VPN config or for a MINIO_ENDPOINT_IP "
        "inside the routed range to pin in /etc/hosts. Note that 'ping' is not a reliable "
        "test — ICMP may be blocked even when the endpoint works."
    )


class Result(NamedTuple):
    name: str
    status: str
    detail: str
    hint: str = ""


def is_placeholder(value: str) -> bool:
    return bool(re.fullmatch(r"x+", value.strip(), re.IGNORECASE))


# Values that are masked in the report unless --show-secrets is given (people paste
# diagnostic output into issues and chat when asking for help).
SECRET_VARS = {"MINIO_SECRET_KEY", "B2_APPLICATION_KEY", "CLEARML_API_SECRET_KEY"}

# Files where the variables are typically defined, in lookup order.
PROFILE_FILES = [
    Path.home() / ".bashrc",
    Path.home() / ".bash_profile",
    Path.home() / ".profile",
    Path.home() / ".zshrc",
    Path.home() / "env_vars.txt",
    Path(__file__).resolve().parents[2] / ".env",
]

VAR_GROUPS = {
    "MinIO": ["MINIO_ENDPOINT_URL", "MINIO_ACCESS_KEY", "MINIO_SECRET_KEY"],
    "Backblaze B2": ["B2_ENDPOINT_URL", "B2_KEY_ID", "B2_APPLICATION_KEY"],
    "ClearML": ["CLEARML_API_HOST", "CLEARML_API_ACCESS_KEY", "CLEARML_API_SECRET_KEY"],
}


def tilde(path: Path) -> str:
    try:
        return "~/" + str(path.relative_to(Path.home()))
    except ValueError:
        return str(path)


def find_definition(var: str) -> str:
    """Return 'file:line' of the first profile file that defines var, or ''."""
    pattern = re.compile(rf"^\s*(?:export\s+)?{re.escape(var)}=")
    for profile in PROFILE_FILES:
        try:
            if not profile.is_file():
                continue
            for line_no, line in enumerate(profile.read_text(errors="replace").splitlines(), 1):
                if pattern.match(line):
                    return f"{tilde(profile)}:{line_no}"
        except OSError:
            continue
    return ""


def display_value(var: str, value: str, show_secrets: bool) -> str:
    if show_secrets or var not in SECRET_VARS:
        return value
    if len(value) <= 8:
        return f"{'*' * len(value)} ({len(value)} chars)"
    return f"{value[:4]}…{value[-4:]} ({len(value)} chars)"


def check_one_var(results: List[Result], group: str, var: str, show_secrets: bool) -> None:
    value = os.getenv(var, "")
    source = find_definition(var)
    if var == "CLEARML_API_HOST" and not value and not source:
        results.append(Result(var, OK, f"not set — defaults to {DEFAULT_CLEARML_API_HOST}"))
    elif value and is_placeholder(value):
        results.append(
            Result(
                var,
                FAIL,
                "placeholder value" + (f" — defined at {source}" if source else ""),
                "Replace the 'xxxx' placeholder from the README template with your real credential.",
            )
        )
    elif value:
        where = f"defined at {source}" if source else "set in this session only (not found in any profile file)"
        results.append(Result(var, OK, f"{display_value(var, value, show_secrets)} — {where}"))
    elif source:
        results.append(
            Result(
                var,
                FAIL,
                f"defined at {source} but not set in the current environment",
                f"Open a new terminal, or run 'source {source.rsplit(':', 1)[0]}' so the export takes effect.",
            )
        )
    else:
        results.append(
            Result(
                var,
                FAIL,
                "not set",
                f"Other {group} variables are set but this one is missing — add 'export {var}=<value>' "
                "to your shell profile (e.g. ~/.bashrc) and open a new terminal.",
            )
        )


def check_env_vars(results: List[Result], show_secrets: bool) -> None:
    data_path = os.getenv("SIL_NLP_DATA_PATH", "")
    source = find_definition("SIL_NLP_DATA_PATH")
    if data_path:
        where = f"defined at {source}" if source else "set in this session only (not found in any profile file)"
        results.append(Result("SIL_NLP_DATA_PATH", OK, f"{data_path} — {where}"))
    elif source:
        results.append(
            Result(
                "SIL_NLP_DATA_PATH",
                FAIL,
                f"defined at {source} but not set in the current environment",
                f"Open a new terminal, or run 'source {source.rsplit(':', 1)[0]}' so the export takes effect.",
            )
        )
    else:
        hint = "Add 'export SIL_NLP_DATA_PATH=<path>' to your shell profile (e.g. ~/.bashrc) and open a new terminal."
        for misspelling in ("SILNLP_DATA_PATH", "SIL_NLP_DATA", "SILNLP_DATA"):
            if os.getenv(misspelling):
                hint = f"{misspelling} is set, but the correct variable name is SIL_NLP_DATA_PATH."
                break
        results.append(Result("SIL_NLP_DATA_PATH", FAIL, "not set", hint))

    for group, variables in VAR_GROUPS.items():
        if not any(os.getenv(v) or find_definition(v) for v in variables):
            if group == "ClearML" and (Path.home() / "clearml.conf").exists():
                results.append(Result("ClearML env vars", OK, "not set — using ~/clearml.conf instead"))
            else:
                results.append(Result(f"{group} env vars", SKIP, "none set", f"Fine if you do not use {group}."))
            continue
        for var in variables:
            check_one_var(results, group, var, show_secrets)


def check_minio_connectivity(results: List[Result]) -> bool:
    """Return True if the MinIO endpoint is fully reachable."""
    url = os.getenv("MINIO_ENDPOINT_URL", "")
    if not url:
        results.append(Result("MinIO connectivity", SKIP, "MINIO_ENDPOINT_URL not set"))
        return False
    parsed = urlparse(url)
    host = parsed.hostname or ""
    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    try:
        ip = socket.getaddrinfo(host, port, socket.AF_INET)[0][4][0]
        results.append(Result("MinIO DNS resolution", OK, f"{host} -> {ip}"))
    except OSError as e:
        results.append(
            Result(
                "MinIO DNS resolution",
                FAIL,
                f"cannot resolve {host} ({e})",
                VPN_DOWN_HINT,
            )
        )
        return False

    try:
        with socket.create_connection((host, port), timeout=TIMEOUT):
            pass
        results.append(Result("MinIO TCP connection", OK, f"{ip}:{port} open"))
    except OSError as e:
        results.append(
            Result(
                "MinIO TCP connection",
                FAIL,
                f"cannot reach {ip}:{port} ({e})",
                VPN_ROUTE_HINT.format(ip=ip),
            )
        )
        return False

    health_url = f"{parsed.scheme}://{host}:{port}/minio/health/live"
    try:
        with urllib.request.urlopen(health_url, timeout=TIMEOUT) as resp:
            status = resp.status
        if status == 200:
            results.append(Result("MinIO health endpoint", OK, "HTTP 200"))
            return True
        results.append(
            Result(
                "MinIO health endpoint",
                FAIL,
                f"HTTP {status}",
                "The server is reachable but unhealthy — likely a server-side problem; contact the SILNLP dev team.",
            )
        )
    except (urllib.error.URLError, OSError) as e:
        results.append(
            Result(
                "MinIO health endpoint",
                FAIL,
                f"request failed ({e})",
                "The port is open but the HTTPS request failed — possibly a TLS or proxy problem.",
            )
        )
    return False


def rclone_config_path() -> Path:
    if os.name == "nt":
        return Path(os.getenv("APPDATA", "")) / "rclone" / "rclone.conf"
    return Path.home() / ".config" / "rclone" / "rclone.conf"


def check_rclone(results: List[Result]) -> None:
    if not shutil.which("rclone"):
        results.append(
            Result(
                "rclone installed",
                FAIL,
                "rclone not found on PATH",
                "Install rclone (https://rclone.org/install/) or run 'source ./rclone_setup.sh minio' from the repo.",
            )
        )
        return
    results.append(Result("rclone installed", OK, shutil.which("rclone") or ""))

    conf_path = rclone_config_path()
    if not conf_path.exists():
        results.append(
            Result(
                "rclone config",
                FAIL,
                f"{conf_path} not found",
                "Copy scripts/rclone/rclone.conf from this repo to that location and fill in your credentials.",
            )
        )
        return

    parser = configparser.ConfigParser()
    try:
        parser.read(conf_path)
    except configparser.Error as e:
        results.append(Result("rclone config", FAIL, f"cannot parse {conf_path} ({e})"))
        return

    if "miniosilnlp" not in parser:
        results.append(
            Result(
                "rclone config",
                WARN,
                "no [miniosilnlp] section",
                "Copy the [miniosilnlp] section from scripts/rclone/rclone.conf if you use MinIO.",
            )
        )
        return

    section = parser["miniosilnlp"]
    problems = []
    hints = []
    for wrong, right in [("access_key", "access_key_id"), ("secret", "secret_access_key")]:
        if wrong in section and right not in section:
            problems.append(f"'{wrong}' should be '{right}'")
    if problems:
        hints.append(
            "Rename the fields in the [miniosilnlp] section — rclone's S3 backend silently ignores unknown option names, which makes authentication fail."
        )
    for field in ("access_key_id", "secret_access_key"):
        if field in section and is_placeholder(section[field]):
            problems.append(f"'{field}' is still a placeholder")
            hints.append(f"Fill in the real value for '{field}'.")

    for field, env_var in [("access_key_id", "MINIO_ACCESS_KEY"), ("secret_access_key", "MINIO_SECRET_KEY")]:
        conf_value = section.get(field, "")
        env_value = os.getenv(env_var, "")
        if conf_value and env_value and conf_value != env_value:
            problems.append(f"'{field}' differs from ${env_var} (lengths {len(conf_value)} vs {len(env_value)})")
            hints.append(
                f"The value of '{field}' in {conf_path} does not match the {env_var} environment "
                "variable — usually a copy/paste error in one of the two. A quick fix: "
                f'sed -i "s|^{field} = .*|{field} = ${env_var}|" {conf_path}'
            )

    if problems:
        results.append(Result("rclone MinIO credentials", FAIL, "; ".join(problems), " ".join(hints)))
    else:
        results.append(Result("rclone MinIO credentials", OK, "field names and values look consistent"))


def check_data_path(results: List[Result]) -> None:
    data_path = os.getenv("SIL_NLP_DATA_PATH", "")
    if not data_path:
        results.append(Result("Data path mount", SKIP, "SIL_NLP_DATA_PATH not set"))
        return
    path = Path(data_path)
    if not path.is_dir():
        results.append(
            Result(
                "Data path mount",
                FAIL,
                f"{path} does not exist or is not a directory",
                f"Create it with 'mkdir -p {path}' and mount the bucket (see bucket_setup.md).",
            )
        )
        return
    try:
        entries = list(path.iterdir())
    except OSError as e:
        results.append(
            Result(
                "Data path mount",
                FAIL,
                f"cannot list {path} ({e})",
                "A stale FUSE mount often causes this. Unmount with "
                f"'fusermount -uz {path}' and re-mount the bucket.",
            )
        )
        return
    mounted = os.path.ismount(path)
    if not entries:
        results.append(
            Result(
                "Data path mount",
                WARN if not mounted else FAIL,
                f"{path} is empty" + (" and not a mount point" if not mounted else ""),
                "If you use MinIO/B2, the bucket is not mounted. Mount it with: rclone mount "
                "--daemon --vfs-cache-mode full --use-server-modtime miniosilnlp:nlp-research "
                f"{path}  (requires the VPN for MinIO). If you store data locally instead, this is fine.",
            )
        )
        return
    kind = "mounted bucket" if mounted else "local directory"
    results.append(Result("Data path mount", OK, f"{path} is a {kind} with {len(entries)} top-level entries"))


def clearml_credentials() -> Tuple[str, str, str, str]:
    """Return (api_host, access_key, secret_key, source)."""
    host = os.getenv("CLEARML_API_HOST", "")
    access = os.getenv("CLEARML_API_ACCESS_KEY", "")
    secret = os.getenv("CLEARML_API_SECRET_KEY", "")
    if access and secret:
        return host or DEFAULT_CLEARML_API_HOST, access, secret, "environment variables"
    conf = Path.home() / "clearml.conf"
    if conf.exists():
        text = conf.read_text(errors="replace")

        def find(pattern: str) -> str:
            m = re.search(pattern + r"""\s*[:=]\s*["']?([^"'\s]+)""", text)
            return m.group(1) if m else ""

        return (
            host or find(r"api_server") or DEFAULT_CLEARML_API_HOST,
            access or find(r"\baccess_key"),
            secret or find(r"\bsecret_key"),
            str(conf),
        )
    return host or DEFAULT_CLEARML_API_HOST, access, secret, ""


def check_clearml(results: List[Result]) -> None:
    api_host, access, secret, source = clearml_credentials()
    if not access or not secret:
        results.append(
            Result(
                "ClearML credentials",
                SKIP,
                "no credentials found",
                "Fine if you do not use ClearML. Otherwise set CLEARML_API_ACCESS_KEY / "
                "CLEARML_API_SECRET_KEY or run 'clearml-init' (see clear_ml_setup.md).",
            )
        )
        return
    auth = base64.b64encode(f"{access}:{secret}".encode()).decode()
    request = urllib.request.Request(f"{api_host.rstrip('/')}/auth.login", headers={"Authorization": f"Basic {auth}"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as resp:
            status = resp.status
    except urllib.error.HTTPError as e:
        status = e.code
    except (urllib.error.URLError, OSError) as e:
        results.append(
            Result(
                "ClearML authentication",
                FAIL,
                f"cannot reach {api_host} ({e})",
                "The ClearML server is on the public internet (no VPN needed) — check your internet connection and the CLEARML_API_HOST value.",
            )
        )
        return
    if status == 200:
        results.append(
            Result("ClearML authentication", OK, f"authenticated against {api_host} (credentials from {source})")
        )
    elif status == 401:
        results.append(
            Result(
                "ClearML authentication",
                FAIL,
                f"credentials rejected by {api_host}",
                f"The credentials (from {source}) are wrong or revoked. Generate new ones in the "
                "ClearML web UI (Settings > Workspace > Create new credentials) and re-run clearml-init.",
            )
        )
    else:
        results.append(Result("ClearML authentication", WARN, f"unexpected HTTP {status} from {api_host}"))


def print_report(results: List[Result], show_secrets: bool) -> int:
    width = max(len(r.name) for r in results)
    print()
    print("SILNLP setup check")
    print(
        "Variables are read from the process environment — normally exported from a shell profile "
        f"({', '.join(tilde(f) for f in PROFILE_FILES if f.is_file())}).\n"
        f"ClearML also falls back to ~/clearml.conf; rclone reads {tilde(rclone_config_path())}."
    )
    print("=" * (width + 50))
    for r in results:
        print(f"{ICONS[r.status]} {r.name.ljust(width)}  {r.detail}")
        if r.hint and r.status in (FAIL, WARN):
            for line in _wrap(r.hint, 100):
                print(f"   {' ' * width}  {line}")
    failures = [r for r in results if r.status == FAIL]
    warnings = [r for r in results if r.status == WARN]
    print("=" * (width + 50))
    print(f"{len(failures)} failure(s), {len(warnings)} warning(s)")
    if not show_secrets:
        print("Secret values are partially masked — run with --show-secrets to print them in full.")
    return 1 if failures else 0


def _wrap(text: str, width: int) -> List[str]:
    words = text.split()
    lines: List[str] = []
    current = ""
    for word in words:
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


def main() -> int:
    arg_parser = argparse.ArgumentParser(description="Diagnose a local SILNLP setup.")
    arg_parser.add_argument(
        "--show-secrets",
        action="store_true",
        help="print full credential values instead of masked ones (avoid pasting that output into shared channels)",
    )
    args = arg_parser.parse_args()

    results: List[Result] = []
    check_env_vars(results, args.show_secrets)
    check_minio_connectivity(results)
    check_rclone(results)
    check_data_path(results)
    check_clearml(results)
    return print_report(results, args.show_secrets)


if __name__ == "__main__":
    sys.exit(main())
