---
name: diagnose-setup
description: Diagnose a local SILNLP environment setup — environment variables, VPN/MinIO connectivity, rclone configuration and mount, and ClearML authentication. Use when setup fails, the M drive / MinIO bucket is unreachable, ClearML gives auth errors, or when onboarding a new machine. Invoke with /diagnose-setup.
---

# Diagnose SILNLP setup

Diagnose why a local SILNLP setup is not working. Supported platforms: Ubuntu/Linux, macOS, and WSL (Ubuntu on Windows). On WSL, remember that the VPN runs on the *Windows host*, not inside WSL.

## Step 1: Run the deterministic checker

```
python3 silnlp/common/check_setup.py
```

It is stdlib-only, so it runs even when the poetry/conda environment is broken. Every failed check prints a remediation hint. The report lists each credential variable individually with its value and where it is defined (e.g. `~/.bashrc:142`); secret values are partially masked — `--show-secrets` prints them in full, but never paste that output into shared channels. If all checks pass but the user still has a problem, ask what command or workflow fails and investigate from there.

Do not stop at the first failure — read the whole report. Failures cascade (e.g. a down VPN fails DNS, TCP, health, and mount checks at once); fix the earliest check in the chain first.

## Step 2: Investigate failures with this playbook

### Environment variables

- The correct name is `SIL_NLP_DATA_PATH` (with underscores) — `SILNLP_DATA_PATH` is a common misspelling the script detects.
- Variables belong in `~/.bashrc` (Linux/WSL) or `~/.profile` (macOS) as `export VAR=value`. A new terminal is needed after editing. Also check the user's `env_vars.txt` if they use `setup_env_vars.sh`.
- Watch for placeholder values (`xxxx...`) copied from the README template and never replaced.

### MinIO unreachable (DNS or TCP check fails) — usually the VPN

Ground rules learned the hard way:

- **`ping` is not a valid test** — ICMP may be blocked while the service works. Test with `curl -s -m 8 -o /dev/null -w "%{http_code}" $MINIO_ENDPOINT_URL/minio/health/live` (expect `200`).
- **DNS resolves but TCP fails** means a *routing* problem, not a name problem: the resolved IP is probably outside the VPN's `AllowedIPs`. Compare `getent hosts <minio-host>` against `sudo wg show wg0 allowed-ips` and `ip route`. Known instance: the server resolves to `172.21.50.181` but a client config routing only `10.0.0.0/8` will silently drop the traffic — the fix is adding a range covering it (e.g. `172.21.50.181/32`) to `AllowedIPs`, or pinning a routable address via `MINIO_ENDPOINT_IP` (written to `/etc/hosts` by `setup_env_vars.sh`).

Identify which mechanism manages the tunnel — there are three, and confusion between them causes most VPN mysteries:

1. **NetworkManager** (`nmcli connection show` lists a wireguard entry). NM keeps its *own copy* of the config at `/etc/NetworkManager/system-connections/` — editing `~/wg0.conf` or `/etc/wireguard/wg0.conf` has **no effect** on it. To apply a new config: `sudo nmcli connection delete wg0 && sudo nmcli connection import type wireguard file ~/wg0.conf && sudo nmcli connection up wg0`.
2. **wg-quick** (`systemctl status wg-quick@wg0`). Reads `/etc/wireguard/wg0.conf` at start time only. To apply a new config: `sudo install -m 600 ~/wg0.conf /etc/wireguard/wg0.conf && sudo systemctl restart wg-quick@wg0`.
3. **Windows host** (WSL only). Check the WireGuard app on Windows; nothing inside WSL manages the tunnel.

Additional traps:

- If *both* NetworkManager and wg-quick are configured, they race for the interface at boot. Keep one; disable the other (`sudo systemctl disable wg-quick@wg0` or delete the NM connection).
- `wg-quick: 'wg0' already exists` when starting means an orphaned interface was left behind (e.g. after deleting an NM connection): `sudo ip link delete wg0`, then restart the service.
- The runtime truth is `sudo wg show wg0 allowed-ips` and `ip route` — not any file. Files only matter at (re)start time.
- `sudo` prompts do not work in non-interactive shells; ask the user to run sudo commands in their own terminal and report back. Never ask them to paste passwords.

### rclone / bucket mount

- The config template is `scripts/rclone/rclone.conf` in this repo; it belongs at `~/.config/rclone/rclone.conf` (Linux/WSL/macOS) or `%APPDATA%\rclone\rclone.conf` (Windows).
- The S3 field names must be exactly `access_key_id` and `secret_access_key`. rclone **silently ignores unknown option names** (like `access_key` or `secret`), producing confusing auth failures.
- Compare config values against `$MINIO_ACCESS_KEY` / `$MINIO_SECRET_KEY` for equality **without printing secrets** (compare lengths or use shell `[ "$a" = "$b" ]`). Copy/paste errors (truncated or doubled secrets) are common.
- Test credentials with `rclone lsd miniosilnlp:` — expect a `nlp-research` bucket listing. `SignatureDoesNotMatch` means the secret key is wrong; `403 AccessDenied` means the access key or permissions.
- Mount: `rclone mount --daemon --vfs-cache-mode full --use-server-modtime miniosilnlp:nlp-research ~/M`. A hanging `ls` or "Transport endpoint is not connected" means a stale FUSE mount: `fusermount -uz ~/M`, then re-mount.
- For a crontab `@reboot` auto-mount, use the **absolute path** to rclone (e.g. `/home/<user>/.local/bin/rclone`) — cron's PATH does not include `~/.local/bin`, so the mount fails silently otherwise.

### ClearML

- The checker authenticates against the API directly with HTTP basic auth, so it needs no poetry environment and no VPN (the ClearML server is on the public internet).
- HTTP 401 means wrong or revoked credentials: regenerate in the ClearML web UI (Settings > Workspace > Create new credentials) and re-run `clearml-init`, or update the `CLEARML_API_*` variables.
- The `clearml` Python package lives in the *poetry* virtualenv, not the conda env — `python -c "import clearml"` failing in the bare conda env is normal. Test with `poetry run python -c "import clearml"`.

### Data path

- `SIL_NLP_DATA_PATH` may legitimately point at a local directory instead of a mounted bucket (for offline use) — an unmounted path is only a failure if the user intends to use MinIO/B2.
- Empty mount point + reachable MinIO = the mount simply was not run. It must be re-run after every reboot unless the user sets up the crontab entry (see `bucket_setup.md`).

## Reporting

Summarize as a status table (item / status / what was found), lead with what is broken and the single next action. When a fix needs `sudo`, give the exact commands for the user to run in their own terminal and verify the result yourself afterwards with non-privileged commands.
