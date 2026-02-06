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

    def is_available(self) -> bool:
        """Check if openclaw CLI is on PATH and config exists."""
        cli_found = shutil.which(self._config.cli_path) is not None
        has_token = bool(self._config.auth_token)
        return cli_found and has_token

    async def _run_cli(self, args: list[str], timeout: float = 60.0) -> OpenClawResponse:
        """
        Run an openclaw CLI command and parse JSON output.

        Args:
            args: CLI arguments (after 'openclaw')
            timeout: Command timeout in seconds

        Returns:
            Parsed OpenClawResponse
        """
        cmd = [self._config.cli_path] + args + ["--json"]
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

            # Parse JSON output
            try:
                data = json.loads(stdout_text) if stdout_text else {}
            except json.JSONDecodeError:
                # If not valid JSON, use raw output
                data = {"result": stdout_text}

            content = json.dumps(data, indent=2) if isinstance(data, dict) else str(data)
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
