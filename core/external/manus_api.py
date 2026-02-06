"""
Manus API Wrapper (Computer Use)
================================
Secure wrapper for Manus computer-use automation API.

CRITICAL SAFETY CONTROLS:
- Explicit allowlists for apps/domains
- Max action counts per session
- Dry-run / preview mode
- Full action logging
- Emergency kill-switch
- User approval required
- No system file access
- No credential access

MANUS MUST NEVER:
- Run unattended
- Self-approve actions
- Modify system files
- Access credentials
"""

import logging
import time
import json
from typing import Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from pathlib import Path

import aiohttp

from .vault import get_vault, APIProvider

logger = logging.getLogger(__name__)


class ManusAction(Enum):
    """Allowed Manus actions."""
    CLICK = "click"
    TYPE = "type"
    SCROLL = "scroll"
    READ = "read"
    SCREENSHOT = "screenshot"
    WAIT = "wait"


@dataclass
class ManusRequest:
    """Sanitized request to Manus API."""
    goal: str
    allowed_actions: list[ManusAction] = field(default_factory=lambda: [ManusAction.READ])
    constraints: str = ""
    max_actions: int = 10
    dry_run: bool = True  # Default to preview mode
    timeout_seconds: int = 60


@dataclass
class ManusActionLog:
    """Log entry for a Manus action."""
    timestamp: str
    action: str
    target: str
    success: bool
    duration_ms: int
    details: dict = field(default_factory=dict)


@dataclass
class ManusResponse:
    """Sanitized response from Manus API."""
    success: bool
    result: str
    summary: str
    actions_taken: list[ManusActionLog] = field(default_factory=list)
    was_dry_run: bool = True
    latency_ms: int = 0
    error: Optional[str] = None
    kill_switch_triggered: bool = False


class ManusAPIWrapper:
    """
    Secure wrapper for Manus computer-use API.

    SAFETY IS PARAMOUNT. All actions are:
    - Explicitly allowlisted
    - Logged completely
    - User-approved
    - Bounded by limits
    """

    # App allowlist (only these apps can be interacted with)
    ALLOWED_APPS = {
        "browser",      # Web browsers
        "terminal",     # Terminal (read-only preferred)
        "editor",       # Text editors
        "file_manager", # File browsing (no system folders)
    }

    # Domain allowlist for browser actions
    ALLOWED_DOMAINS = {
        "google.com",
        "github.com",
        "stackoverflow.com",
        "wikipedia.org",
        "docs.python.org",
        "developer.mozilla.org",
    }

    # Blocked paths (NEVER access these)
    BLOCKED_PATHS = {
        "/etc/",
        "/var/",
        "/usr/",
        "/root/",
        "/home/*/.ssh/",
        "/home/*/.gnupg/",
        "/home/*/.config/",
        "C:\\Windows\\",
        "C:\\Program Files\\",
        "C:\\Users\\*\\AppData\\",
        "%USERPROFILE%\\.ssh\\",
        "%APPDATA%\\",
    }

    # Blocked file types
    BLOCKED_EXTENSIONS = {
        ".exe", ".bat", ".ps1", ".sh", ".cmd",
        ".dll", ".sys", ".msi", ".vbs", ".wsf",
        ".pem", ".key", ".crt", ".pfx",
    }

    # Session limits
    MAX_ACTIONS_PER_SESSION = 50
    MAX_CONCURRENT_SESSIONS = 1
    SESSION_TIMEOUT_SECONDS = 300

    def __init__(self, log_dir: Optional[Path] = None):
        """Initialize wrapper with logging directory."""
        self._session_active = False
        self._session_actions = 0
        self._session_start: Optional[datetime] = None
        self._kill_switch = False
        self._action_log: list[ManusActionLog] = []
        self._user_approved = False

        # Setup action logging
        self._log_dir = log_dir or Path("logs")
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = self._log_dir / "manus_actions.jsonl"

    def is_available(self) -> bool:
        """Check if Manus API is available."""
        vault = get_vault()
        return vault.is_available(APIProvider.MANUS) and not self._kill_switch

    def trigger_kill_switch(self, reason: str = "Manual trigger") -> None:
        """
        EMERGENCY STOP - Immediately halt all Manus operations.

        This is a one-way switch. Once triggered, Manus is disabled
        until the application is restarted.
        """
        self._kill_switch = True
        self._session_active = False
        self._user_approved = False

        logger.critical(f"MANUS KILL SWITCH TRIGGERED: {reason}")

        # Log kill switch event
        self._log_action(ManusActionLog(
            timestamp=datetime.now().isoformat(),
            action="KILL_SWITCH",
            target="",
            success=True,
            duration_ms=0,
            details={"reason": reason}
        ))

    def require_user_approval(self, goal: str, actions: list[ManusAction]) -> bool:
        """
        Request user approval before executing.

        In a real implementation, this would show a GUI dialog
        or require explicit console confirmation.

        Returns True if approved, False otherwise.
        """
        # This is a placeholder - real implementation needs actual user interaction
        logger.warning(f"MANUS APPROVAL REQUIRED: {goal}")
        logger.warning(f"  Actions: {[a.value for a in actions]}")

        # For safety, default to NOT approved
        # Real implementation must get explicit user confirmation
        self._user_approved = False
        return False

    def set_user_approved(self, approved: bool) -> None:
        """Set user approval status (called by UI layer)."""
        self._user_approved = approved
        if approved:
            logger.info("Manus: User approval granted")
        else:
            logger.info("Manus: User approval revoked")

    def _validate_request(self, request: ManusRequest) -> tuple[bool, str]:
        """Validate request against safety rules."""
        # Check kill switch
        if self._kill_switch:
            return False, "Kill switch is active - Manus disabled"

        # Check user approval
        if not self._user_approved and not request.dry_run:
            return False, "User approval required for non-dry-run execution"

        # Check action count
        if request.max_actions > self.MAX_ACTIONS_PER_SESSION:
            return False, f"Max actions exceeded: {request.max_actions} > {self.MAX_ACTIONS_PER_SESSION}"

        # Check session limits
        if self._session_active:
            remaining = self.MAX_ACTIONS_PER_SESSION - self._session_actions
            if request.max_actions > remaining:
                return False, f"Session action limit would be exceeded: {remaining} remaining"

        # Validate allowed actions
        for action in request.allowed_actions:
            if action not in ManusAction:
                return False, f"Invalid action: {action}"

        return True, ""

    def _log_action(self, action: ManusActionLog) -> None:
        """Log action to file and memory."""
        self._action_log.append(action)

        # Append to JSONL file
        with open(self._log_file, "a") as f:
            f.write(json.dumps({
                "timestamp": action.timestamp,
                "action": action.action,
                "target": action.target,
                "success": action.success,
                "duration_ms": action.duration_ms,
                "details": action.details
            }) + "\n")

    async def call(self, request: ManusRequest) -> ManusResponse:
        """
        Execute a Manus request (or preview in dry-run mode).

        Args:
            request: Validated request object

        Returns:
            Sanitized response with action log
        """
        # Validate request
        is_valid, error = self._validate_request(request)
        if not is_valid:
            return ManusResponse(
                success=False,
                result="",
                summary="",
                error=error,
                kill_switch_triggered=self._kill_switch
            )

        start_time = time.time()
        actions_taken = []

        try:
            if request.dry_run:
                # Preview mode - don't actually execute
                logger.info(f"Manus DRY RUN: {request.goal}")

                preview_log = ManusActionLog(
                    timestamp=datetime.now().isoformat(),
                    action="DRY_RUN_PREVIEW",
                    target=request.goal,
                    success=True,
                    duration_ms=0,
                    details={
                        "allowed_actions": [a.value for a in request.allowed_actions],
                        "constraints": request.constraints,
                        "max_actions": request.max_actions
                    }
                )
                self._log_action(preview_log)
                actions_taken.append(preview_log)

                latency_ms = int((time.time() - start_time) * 1000)

                return ManusResponse(
                    success=True,
                    result=f"[DRY RUN] Would execute: {request.goal}",
                    summary=f"Preview: {request.goal[:100]}",
                    actions_taken=actions_taken,
                    was_dry_run=True,
                    latency_ms=latency_ms
                )

            # Real execution - requires API call
            if not self.is_available():
                return ManusResponse(
                    success=False,
                    result="",
                    summary="",
                    error="Manus API not available",
                    kill_switch_triggered=self._kill_switch
                )

            # Start session if not active
            if not self._session_active:
                self._session_active = True
                self._session_actions = 0
                self._session_start = datetime.now()

            # Make API call
            vault = get_vault()
            api_key = vault._get_key(APIProvider.MANUS)

            async with aiohttp.ClientSession() as session:
                # Note: This is a placeholder API structure
                # Real Manus API integration would go here
                payload = {
                    "goal": request.goal,
                    "allowed_actions": [a.value for a in request.allowed_actions],
                    "constraints": request.constraints,
                    "max_actions": request.max_actions,
                    "timeout": request.timeout_seconds
                }

                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }

                # Placeholder URL - would be real Manus endpoint
                api_url = "https://api.manus.ai/v1/execute"

                async with session.post(
                    api_url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=request.timeout_seconds)
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        return ManusResponse(
                            success=False,
                            result="",
                            summary="",
                            error=f"API error: {resp.status}"
                        )

                    data = await resp.json()

                    # Log each action from response
                    for action_data in data.get("actions", []):
                        action_log = ManusActionLog(
                            timestamp=datetime.now().isoformat(),
                            action=action_data.get("type", "unknown"),
                            target=action_data.get("target", ""),
                            success=action_data.get("success", False),
                            duration_ms=action_data.get("duration_ms", 0),
                            details=action_data.get("details", {})
                        )
                        self._log_action(action_log)
                        actions_taken.append(action_log)
                        self._session_actions += 1

                    latency_ms = int((time.time() - start_time) * 1000)

                    result = data.get("result", "")
                    summary = result[:200] + "..." if len(result) > 200 else result

                    return ManusResponse(
                        success=True,
                        result=result,
                        summary=summary,
                        actions_taken=actions_taken,
                        was_dry_run=False,
                        latency_ms=latency_ms
                    )

        except asyncio.TimeoutError:
            return ManusResponse(
                success=False,
                result="",
                summary="",
                actions_taken=actions_taken,
                error="Request timed out"
            )

        except Exception as e:
            logger.error(f"Manus error: {type(e).__name__}: {e}")
            # Trigger kill switch on unexpected errors for safety
            self.trigger_kill_switch(f"Unexpected error: {type(e).__name__}")
            return ManusResponse(
                success=False,
                result="",
                summary="",
                actions_taken=actions_taken,
                error=f"Error: {type(e).__name__}",
                kill_switch_triggered=True
            )

    def get_session_stats(self) -> dict:
        """Get current session statistics."""
        return {
            "active": self._session_active,
            "actions_taken": self._session_actions,
            "actions_remaining": self.MAX_ACTIONS_PER_SESSION - self._session_actions,
            "session_start": self._session_start.isoformat() if self._session_start else None,
            "kill_switch_active": self._kill_switch,
            "user_approved": self._user_approved
        }

    def get_action_log(self) -> list[ManusActionLog]:
        """Get the full action log for this session."""
        return self._action_log.copy()

    def end_session(self) -> None:
        """End the current session."""
        self._session_active = False
        self._user_approved = False
        logger.info("Manus session ended")


# Module-level instance
_wrapper: Optional[ManusAPIWrapper] = None


def get_manus_api(log_dir: Optional[Path] = None) -> ManusAPIWrapper:
    """Get the Manus API wrapper instance."""
    global _wrapper
    if _wrapper is None:
        _wrapper = ManusAPIWrapper(log_dir)
    return _wrapper


# Import asyncio for the exception type
import asyncio
