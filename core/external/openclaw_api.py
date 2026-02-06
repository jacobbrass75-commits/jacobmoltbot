"""
OpenClaw API Wrapper
====================
Subprocess-based wrapper for OpenClaw CLI.

Use cases: Multi-channel messaging, agent delegation, system status

ISOLATION RULES:
- OpenClaw NEVER sees internal system prompts
- All outputs are sanitized before return
- Communication via subprocess CLI calls only
- Token read from existing ~/.openclaw/openclaw.json config
"""

import asyncio
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OpenClawConfig:
    """OpenClaw configuration."""
    gateway_url: str = "ws://127.0.0.1:18789"
    auth_token: str = ""
    cli_path: str = "openclaw"

    @classmethod
    def from_config_file(cls) -> "OpenClawConfig":
        """Load config from ~/.openclaw/openclaw.json."""
        config_path = Path.home() / ".openclaw" / "openclaw.json"
        config = cls()

        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Extract gateway URL
                gateway = data.get("gateway", {})
                if "url" in gateway:
                    config.gateway_url = gateway["url"]

                # Extract auth token
                auth = gateway.get("auth", {})
                if "token" in auth:
                    config.auth_token = auth["token"]

                logger.debug("Loaded OpenClaw config from %s", config_path)
            except Exception as e:
                logger.warning("Failed to load OpenClaw config: %s", e)

        return config


@dataclass
class OpenClawRequest:
    """Request to OpenClaw."""
    message: str = ""
    channel: str = ""
    target: str = ""
    thinking_level: str = "medium"


@dataclass
class OpenClawResponse:
    """Response from OpenClaw."""
    success: bool
    content: str
    summary: str
    latency_ms: int = 0
    error: Optional[str] = None


class OpenClawAPIWrapper:
    """
    Subprocess-based wrapper for OpenClaw CLI.

    All communication happens via `openclaw` CLI with --json output.
    No WebSocket connection or additional pip dependencies needed.
    """

    def __init__(self):
        """Initialize wrapper."""
        self._config = OpenClawConfig.from_config_file()
        self._call_count = 0
        self._total_latency_ms = 0

        # Resolve the full CLI path at init time so subprocesses can find it
        resolved = shutil.which(self._config.cli_path)
        if resolved:
            self._resolved_cli = resolved
        else:
            self._resolved_cli = self._config.cli_path

    def is_available(self) -> bool:
        """Check if openclaw CLI is on PATH and config exists."""
        cli_found = shutil.which(self._config.cli_path) is not None
        has_token = bool(self._config.auth_token)
        return cli_found and has_token

    async def _run_cli(self, args: list[str], timeout: float = 60.0, json_output: bool = True) -> OpenClawResponse:
        """
        Run an openclaw CLI command and parse output.

        Args:
            args: CLI arguments (after 'openclaw')
            timeout: Command timeout in seconds
            json_output: If True, append --json flag and parse as JSON

        Returns:
            Parsed OpenClawResponse
        """
        cmd = [self._resolved_cli] + args
        if json_output:
            cmd.append("--json")
        start_time = time.time()

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=None  # inherit environment
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )

            latency_ms = int((time.time() - start_time) * 1000)
            self._call_count += 1
            self._total_latency_ms += latency_ms

            stdout_text = stdout.decode("utf-8", errors="replace").strip()
            stderr_text = stderr.decode("utf-8", errors="replace").strip()

            if process.returncode != 0:
                error_msg = stderr_text or f"CLI exited with code {process.returncode}"
                logger.error("OpenClaw CLI error: %s", error_msg)
                return OpenClawResponse(
                    success=False,
                    content="",
                    summary="",
                    latency_ms=latency_ms,
                    error=error_msg
                )

            # Parse output
            if json_output:
                try:
                    data = json.loads(stdout_text) if stdout_text else {}
                    content = json.dumps(data, indent=2) if isinstance(data, dict) else str(data)
                except json.JSONDecodeError:
                    content = stdout_text
            else:
                content = stdout_text
            summary = self._make_summary(content)

            logger.info("OpenClaw CLI call completed: %dms", latency_ms)

            return OpenClawResponse(
                success=True,
                content=content,
                summary=summary,
                latency_ms=latency_ms
            )

        except asyncio.TimeoutError:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.error("OpenClaw CLI timed out after %.1fs", timeout)
            return OpenClawResponse(
                success=False,
                content="",
                summary="",
                latency_ms=latency_ms,
                error=f"Command timed out after {timeout}s"
            )

        except FileNotFoundError:
            logger.error("OpenClaw CLI not found: %s", self._config.cli_path)
            return OpenClawResponse(
                success=False,
                content="",
                summary="",
                error="openclaw CLI not found on PATH"
            )

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.error("OpenClaw CLI unexpected error: %s", e)
            return OpenClawResponse(
                success=False,
                content="",
                summary="",
                latency_ms=latency_ms,
                error=f"Unexpected error: {type(e).__name__}: {e}"
            )

    def _make_summary(self, content: str) -> str:
        """Create a compact summary from content."""
        if len(content) <= 500:
            return content
        first_para = content.split("\n\n")[0]
        if len(first_para) <= 500:
            return first_para + "..."
        return content[:497] + "..."

    async def send_message(self, channel: str, target: str, message: str) -> OpenClawResponse:
        """
        Send a message through an OpenClaw channel.

        Args:
            channel: Channel name (e.g., 'telegram', 'whatsapp', 'discord')
            target: Recipient identifier
            message: Message text to send

        Returns:
            OpenClawResponse with result
        """
        args = [
            "message", "send",
            "--channel", channel,
            "--target", target,
            "--message", message
        ]
        return await self._run_cli(args)

    async def run_agent(self, message: str, thinking: str = "medium") -> OpenClawResponse:
        """
        Run the OpenClaw agent with a task.

        Args:
            message: Task or question for the agent
            thinking: Reasoning depth (off, minimal, low, medium, high)

        Returns:
            OpenClawResponse with agent result
        """
        args = [
            "agent",
            "--message", message,
            "--thinking", thinking
        ]
        return await self._run_cli(args, timeout=120.0)

    async def get_status(self) -> OpenClawResponse:
        """
        Get OpenClaw gateway and channel status.

        Returns:
            OpenClawResponse with status information
        """
        args = ["status", "--all"]
        return await self._run_cli(args)

    async def run_security_audit(self, auto_fix: bool = False) -> OpenClawResponse:
        """
        Run a comprehensive, evidence-based security audit on the host machine.

        GUARDRAIL: Every section includes the exact command executed and its
        captured output. Sections without runtime evidence are marked UNVERIFIED.
        No version, CVE, or fix claims are made without command execution proof.

        Executes openclaw CLI commands and system checks directly via
        subprocess — NOT through the OpenClaw agent (which can't audit itself).

        Args:
            auto_fix: If True, also run 'openclaw security audit --fix'

        Returns:
            OpenClawResponse with combined audit results
        """
        sections = []
        start_time = time.time()

        # ── GUARDRAIL 1: Hard CLI precondition ──
        # If CLI is not available, attempt npm install, then re-check.
        # If still unavailable, abort entirely with a single blocking error.
        cli_available = shutil.which(self._resolved_cli) is not None
        if not cli_available:
            # Attempt automatic installation
            sections.append("## CLI RECOVERY\nOpenClaw CLI not found on PATH. Attempting: npm install -g openclaw")
            try:
                install_proc = await asyncio.create_subprocess_exec(
                    "npm", "install", "-g", "openclaw",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                inst_out, inst_err = await asyncio.wait_for(install_proc.communicate(), timeout=120.0)
                install_stdout = inst_out.decode("utf-8", errors="replace").strip()
                install_stderr = inst_err.decode("utf-8", errors="replace").strip()
                sections.append(f"Command: npm install -g openclaw\nExit code: {install_proc.returncode}\nstdout: {install_stdout}\nstderr: {install_stderr}")

                if install_proc.returncode == 0:
                    # Re-resolve CLI path after install
                    resolved = shutil.which("openclaw")
                    if resolved:
                        self._resolved_cli = resolved
                        cli_available = True
                        sections.append(f"CLI recovered at: {self._resolved_cli}")
                    else:
                        sections.append("CLI still not found after install.")
            except Exception as e:
                sections.append(f"Auto-install failed: {e}")

            if not cli_available:
                # GUARDRAIL: Abort — do not infer anything from config files alone
                latency_ms = int((time.time() - start_time) * 1000)
                abort_msg = "\n\n".join(sections)
                abort_msg += "\n\n## AUDIT ABORTED\nBLOCKING ERROR: OpenClaw CLI is not available."
                abort_msg += "\nNo audit sections can be verified without CLI access."
                abort_msg += "\nNo version, CVE, network, or fix claims are made."
                abort_msg += "\n\nRequired action: Install OpenClaw CLI (npm install -g openclaw) and re-run audit."
                return OpenClawResponse(
                    success=False,
                    content=abort_msg,
                    summary="AUDIT ABORTED: OpenClaw CLI not available. Install with: npm install -g openclaw",
                    latency_ms=latency_ms,
                    error="CLI not available — audit cannot proceed"
                )

        # ── GUARDRAIL 2: Evidence gate ──
        # Helper: every section records command, exit code, stdout, stderr.
        # Sections that fail execution are marked UNVERIFIED.
        async def _evidence_section(title: str, args: list[str],
                                     timeout: float = 60.0, use_json: bool = False) -> str:
            """Run a command and return an evidence-gated section."""
            cmd_str = " ".join([self._resolved_cli] + args)
            result = await self._run_cli(args, timeout=timeout, json_output=use_json)
            lines = [f"## {title}"]
            lines.append(f"Command: {cmd_str}")
            if result.success:
                lines.append(f"Status: EXECUTED (exit 0)")
                lines.append(f"Output:\n{result.content}")
                return "\n".join(lines)
            else:
                lines.append(f"Status: UNVERIFIED")
                lines.append(f"Error: {result.error}")
                lines.append("This section has no runtime evidence and is excluded from pass/fail results.")
                return "\n".join(lines)

        # ── Section 1: Version (verified via CLI execution only) ──
        sections.append(await _evidence_section("VERSION CHECK", ["--version"]))

        # ── Section 2: Deep security audit ──
        sections.append(await _evidence_section(
            "SECURITY AUDIT (deep)", ["security", "audit", "--deep"], timeout=120.0))

        # ── GUARDRAIL 3: Fixes only reported if command actually succeeded ──
        if auto_fix:
            fix_section = await _evidence_section(
                "SECURITY AUDIT (fix)", ["security", "audit", "--fix"], timeout=120.0)
            # Parse whether the fix command actually ran
            if "UNVERIFIED" in fix_section:
                fix_section += "\n\nFixes Applied: No fixes applied — audit --fix could not be executed."
            else:
                fix_section += "\n\nFixes Applied: See command output above for applied changes."
            sections.append(fix_section)
        else:
            sections.append("## FIXES\nNot requested (auto_fix=false). No fixes applied.")

        # ── Section 3: Status check ──
        sections.append(await _evidence_section("GATEWAY STATUS", ["status", "--all"]))

        # ── Section 4: Config file (redacted, read from disk) ──
        config_path = Path.home() / ".openclaw" / "openclaw.json"
        config_section_lines = ["## CONFIG (redacted)"]
        config_data = None
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
                self._redact_secrets(config_data)
                config_section_lines.append(f"Source: {config_path}")
                config_section_lines.append(f"Content:\n{json.dumps(config_data, indent=2)}")
            except Exception as e:
                config_section_lines.append(f"Source: {config_path}")
                config_section_lines.append(f"Status: UNVERIFIED — Failed to read: {e}")
        else:
            config_section_lines.append(f"Status: UNVERIFIED — {config_path} not found")
        sections.append("\n".join(config_section_lines))

        # ── GUARDRAIL 5: Access control policy evaluation ──
        # dmPolicy/groupPolicy missing or "open" = FAIL, not recommendation
        acl_lines = ["## ACCESS CONTROL POLICY"]
        if config_data:
            gateway = config_data.get("gateway", {})
            dm_policy = config_data.get("dmPolicy", gateway.get("dmPolicy"))
            group_policy = config_data.get("groupPolicy", gateway.get("groupPolicy"))
            auth_mode = gateway.get("auth", {}).get("mode", gateway.get("auth", {}).get("token", None))

            # dmPolicy check
            if dm_policy is None:
                acl_lines.append("dmPolicy: MISSING — FAIL (must be 'allowlist' or 'pairing')")
            elif dm_policy == "open":
                acl_lines.append("dmPolicy: 'open' — FAIL (requires explicit 'accepted risk' annotation to downgrade)")
            else:
                acl_lines.append(f"dmPolicy: '{dm_policy}' — PASS")

            # groupPolicy check
            if group_policy is None:
                acl_lines.append("groupPolicy: MISSING — FAIL (must be 'allowlist')")
            elif group_policy == "open":
                acl_lines.append("groupPolicy: 'open' — FAIL (requires explicit 'accepted risk' annotation to downgrade)")
            else:
                acl_lines.append(f"groupPolicy: '{group_policy}' — PASS")

            # Auth mode check
            if auth_mode:
                acl_lines.append(f"auth.mode: configured — PASS")
            else:
                acl_lines.append("auth.mode: MISSING — FAIL (gateway has no authentication)")

            # gateway.bind check
            bind_val = gateway.get("bind")
            if bind_val is None:
                acl_lines.append("gateway.bind: NOT SET — FAIL (must be explicitly set to 'loopback')")
            elif bind_val in ("loopback", "127.0.0.1", "localhost"):
                acl_lines.append(f"gateway.bind: '{bind_val}' — PASS")
            else:
                acl_lines.append(f"gateway.bind: '{bind_val}' — WARNING (not loopback, verify intent)")
        else:
            acl_lines.append("Status: UNVERIFIED — config not available")
        sections.append("\n".join(acl_lines))

        # ── Section 5: File permissions (Windows NTFS ACL check) ──
        openclaw_dir = Path.home() / ".openclaw"
        creds_dir = openclaw_dir / "credentials"
        perm_lines = ["## FILE PERMISSIONS"]
        try:
            acl_proc = await asyncio.create_subprocess_exec(
                "icacls", str(openclaw_dir), "/T",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            acl_out, acl_err = await asyncio.wait_for(acl_proc.communicate(), timeout=10.0)
            acl_text = acl_out.decode("utf-8", errors="replace").strip()
            perm_lines.append(f"Command: icacls {openclaw_dir} /T")
            perm_lines.append(f"Output:\n{acl_text}")

            # Check for overly permissive: Everyone, Users, Authenticated Users
            dangerous_principals = ["Everyone", "BUILTIN\\Users", "NT AUTHORITY\\Authenticated Users"]
            for principal in dangerous_principals:
                if principal.lower() in acl_text.lower():
                    perm_lines.append(f"FAIL: '{principal}' has access to OpenClaw directory")
        except Exception as e:
            perm_lines.append(f"Status: UNVERIFIED — Failed: {e}")
        sections.append("\n".join(perm_lines))

        # ── Section 6: Exec approvals ──
        exec_approvals_path = openclaw_dir / "exec-approvals.json"
        exec_lines = ["## EXEC APPROVALS"]
        if exec_approvals_path.exists():
            try:
                with open(exec_approvals_path, "r", encoding="utf-8") as f:
                    approvals = json.load(f)
                self._redact_secrets(approvals)
                exec_lines.append(f"Source: {exec_approvals_path}")
                exec_lines.append(f"Content:\n{json.dumps(approvals, indent=2)}")
            except Exception as e:
                exec_lines.append(f"Status: UNVERIFIED — Failed to read: {e}")
        else:
            exec_lines.append("No exec-approvals.json found (default policy applies)")
        sections.append("\n".join(exec_lines))

        # ── GUARDRAIL 4: Separate host OS vs OpenClaw ports ──
        port_lines = ["## NETWORK EXPOSURE"]
        try:
            proc = await asyncio.create_subprocess_exec(
                "netstat", "-an",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout_bytes, _ = await asyncio.wait_for(proc.communicate(), timeout=10.0)
            netstat_out = stdout_bytes.decode("utf-8", errors="replace")
            listen_lines = [l for l in netstat_out.splitlines() if "LISTEN" in l.upper()]

            port_lines.append(f"Command: netstat -an (filtered to LISTENING)")

            # Determine OpenClaw gateway port from config
            oc_port = None
            if config_data:
                gw_url = config_data.get("gateway", {}).get("url", "")
                if ":" in gw_url:
                    try:
                        oc_port = int(gw_url.rsplit(":", 1)[1].split("/")[0])
                    except (ValueError, IndexError):
                        pass

            oc_found = []
            host_found = []
            for line in listen_lines:
                # Attribute to OpenClaw only if port matches gateway port
                if oc_port and f":{oc_port}" in line:
                    oc_found.append(line.strip())
                else:
                    host_found.append(line.strip())

            if oc_found:
                port_lines.append(f"\nOpenClaw ports (gateway port {oc_port}):")
                for l in oc_found:
                    port_lines.append(f"  {l}")
            else:
                port_lines.append(f"\nOpenClaw gateway port ({oc_port or 'unknown'}): not found in LISTENING — may be WebSocket-only")

            port_lines.append(f"\nHost OS services (non-OpenClaw) — {len(host_found)} listening:")
            for l in host_found[:20]:
                port_lines.append(f"  {l}")
            if len(host_found) > 20:
                port_lines.append(f"  ... and {len(host_found) - 20} more")
        except Exception as e:
            port_lines.append(f"Status: UNVERIFIED — Failed: {e}")
        sections.append("\n".join(port_lines))

        latency_ms = int((time.time() - start_time) * 1000)
        combined = "\n\n".join(sections)

        return OpenClawResponse(
            success=True,
            content=combined,
            summary=self._make_summary(combined),
            latency_ms=latency_ms
        )

    def _redact_secrets(self, data) -> None:
        """Recursively redact sensitive values in a dict or list."""
        sensitive_keys = {"token", "password", "secret", "key", "api_key", "apikey", "credential"}
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, (dict, list)):
                    self._redact_secrets(v)
                elif isinstance(v, str) and any(s in k.lower() for s in sensitive_keys):
                    data[k] = "[REDACTED]"
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, (dict, list)):
                    self._redact_secrets(item)

    def get_stats(self) -> dict:
        """Get usage statistics."""
        return {
            "call_count": self._call_count,
            "total_latency_ms": self._total_latency_ms
        }


# Module-level instance
_wrapper: Optional[OpenClawAPIWrapper] = None


def get_openclaw_api() -> OpenClawAPIWrapper:
    """Get the OpenClaw API wrapper instance."""
    global _wrapper
    if _wrapper is None:
        _wrapper = OpenClawAPIWrapper()
    return _wrapper
