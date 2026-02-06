"""
Moltbot Authority Policy
========================
Codified rules for autonomous action. Every tool that modifies state
or runs a subprocess must check this module before proceeding.
"""

import logging
import re
from datetime import datetime, timezone
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Categories of actions the bot can take."""
    AUDIT = auto()
    INSTALL_DEPENDENCY = auto()
    CONFIG_WRITE = auto()
    SERVICE_START = auto()
    SERVICE_STOP = auto()
    TTS_OUTPUT = auto()
    FILE_WRITE_BOT_SCOPE = auto()
    LOG_WRITE = auto()
    FILE_DELETE = auto()
    IMPERSONATE = auto()
    EXTERNAL_CALL = auto()
    DESTRUCTIVE_ACTION = auto()
    SEND_MONEY = auto()
    ACCOUNT_CREATION = auto()
    SHELL_COMMAND = auto()


@dataclass
class AuthorityDecision:
    """Result of an authority check."""
    allowed: bool
    reason: str
    action_type: ActionType
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class AuthorityPolicy:
    """
    Central policy engine. Determines what the bot can do autonomously
    vs. what requires user confirmation.

    Fails closed: unknown action types are denied.
    """

    # Actions the bot can perform without asking
    AUTO_ALLOWED: set[ActionType] = {
        ActionType.AUDIT,
        ActionType.INSTALL_DEPENDENCY,
        ActionType.CONFIG_WRITE,
        ActionType.SERVICE_START,
        ActionType.SERVICE_STOP,
        ActionType.TTS_OUTPUT,
        ActionType.FILE_WRITE_BOT_SCOPE,
        ActionType.LOG_WRITE,
        ActionType.SHELL_COMMAND,
    }

    # Actions that must be confirmed by the user before executing
    REQUIRES_CONFIRMATION: set[ActionType] = {
        ActionType.IMPERSONATE,
        ActionType.EXTERNAL_CALL,
        ActionType.DESTRUCTIVE_ACTION,
        ActionType.FILE_DELETE,
        ActionType.SEND_MONEY,
        ActionType.ACCOUNT_CREATION,
    }

    # Shell command allowlist — prefixes that are safe to run
    _SHELL_ALLOWLIST: list[str] = [
        "npm install",
        "npm list",
        "npm --version",
        "npm view",
        "npm outdated",
        "npm update",
        "pip install",
        "pip list",
        "pip show",
        "pip --version",
        "pip freeze",
        "python -m pip",
        "python --version",
        "node --version",
        "openclaw",
        "netstat",
        "ipconfig",
        "ifconfig",
        "ping",
        "nslookup",
        "tracert",
        "traceroute",
        "curl",
        "wget",
        "icacls",
        "git status",
        "git log",
        "git diff",
        "git branch",
        "git remote",
        "git show",
        "git tag",
        "systeminfo",
        "tasklist",
        "whoami",
        "hostname",
        "dir",
        "ls",
        "type",
        "cat",
        "echo",
        "where",
        "which",
        "winget list",
        "winget show",
        "winget search",
        "winget install",
        "schtasks /Query",
        "schtasks /Create",
        "sc query",
    ]

    # Shell command blocklist — patterns that are never allowed
    _SHELL_BLOCKLIST: list[str] = [
        r"rm\s+-rf\s+/",
        r"del\s+/s\s+/q",
        r"\bformat\b",
        r"\bshutdown\b",
        r"\breboot\b",
        r"powershell\s+-enc",
        r"powershell\s+-e\s",
        r"powershell.*-ExecutionPolicy\s+Bypass",
        r"Invoke-Expression",
        r"IEX\s*\(",
        r"DownloadString",
        r"\brd\s+/s",
        r"\brmdir\s+/s",
        r"reg\s+delete",
        r"reg\s+add.*\\Run",
        r"net\s+user\s+.*\s+/add",
        r"net\s+localgroup\s+administrators",
        r"takeown\s+/f",
        r"cipher\s+/w",
        r"diskpart",
        r"bcdedit",
        r"wmic\s+.*delete",
        r"schtasks\s+/Delete",
        r"sc\s+delete",
        r"sc\s+stop",
        r"taskkill\s+/f\s+/im",
        r"mklink",
        r">\s*\\\\",
        r"\|\s*nc\s",
        r"base64\s+-d.*\|\s*sh",
        r"eval\s*\(",
    ]

    def __init__(self):
        self._decision_log: list[AuthorityDecision] = []

    def is_allowed(self, action_type: ActionType) -> tuple[bool, str]:
        """
        Check if an action type is allowed autonomously.

        Returns:
            (allowed, reason) tuple. Fails closed on unknown types.
        """
        if action_type in self.AUTO_ALLOWED:
            decision = AuthorityDecision(
                allowed=True,
                reason=f"{action_type.name} is auto-allowed by policy",
                action_type=action_type,
            )
            self._decision_log.append(decision)
            logger.debug(f"Authority ALLOWED: {action_type.name}")
            return True, decision.reason

        if action_type in self.REQUIRES_CONFIRMATION:
            decision = AuthorityDecision(
                allowed=False,
                reason=f"{action_type.name} requires user confirmation",
                action_type=action_type,
            )
            self._decision_log.append(decision)
            logger.info(f"Authority DENIED (needs confirmation): {action_type.name}")
            return False, decision.reason

        # Fail closed: unknown action types are denied
        decision = AuthorityDecision(
            allowed=False,
            reason=f"Unknown action type {action_type.name} — denied by fail-closed policy",
            action_type=action_type,
        )
        self._decision_log.append(decision)
        logger.warning(f"Authority DENIED (unknown): {action_type.name}")
        return False, decision.reason

    def is_shell_command_allowed(self, command: str) -> tuple[bool, str]:
        """
        Check if a specific shell command is allowed.

        Uses allowlist + blocklist. Blocklist takes precedence.

        Returns:
            (allowed, reason) tuple.
        """
        cmd_stripped = command.strip()

        # Blocklist check first (takes precedence)
        for pattern in self._SHELL_BLOCKLIST:
            if re.search(pattern, cmd_stripped, re.IGNORECASE):
                decision = AuthorityDecision(
                    allowed=False,
                    reason=f"Command blocked by pattern: {pattern}",
                    action_type=ActionType.SHELL_COMMAND,
                )
                self._decision_log.append(decision)
                logger.warning(f"Shell BLOCKED: '{cmd_stripped[:80]}' matched blocklist pattern '{pattern}'")
                return False, decision.reason

        # Allowlist check
        for prefix in self._SHELL_ALLOWLIST:
            if cmd_stripped.lower().startswith(prefix.lower()):
                decision = AuthorityDecision(
                    allowed=True,
                    reason=f"Command allowed by prefix: {prefix}",
                    action_type=ActionType.SHELL_COMMAND,
                )
                self._decision_log.append(decision)
                logger.debug(f"Shell ALLOWED: '{cmd_stripped[:80]}' matched prefix '{prefix}'")
                return True, decision.reason

        # Not on allowlist — deny
        decision = AuthorityDecision(
            allowed=False,
            reason=f"Command not on allowlist: '{cmd_stripped[:80]}'",
            action_type=ActionType.SHELL_COMMAND,
        )
        self._decision_log.append(decision)
        logger.info(f"Shell DENIED (not allowlisted): '{cmd_stripped[:80]}'")
        return False, decision.reason

    def get_decision_log(self) -> list[AuthorityDecision]:
        """Return the full decision log."""
        return list(self._decision_log)

    def get_recent_decisions(self, count: int = 20) -> list[AuthorityDecision]:
        """Return the most recent decisions."""
        return list(self._decision_log[-count:])


# Singleton
_instance: Optional[AuthorityPolicy] = None


def get_authority_policy() -> AuthorityPolicy:
    """Get the singleton AuthorityPolicy instance."""
    global _instance
    if _instance is None:
        _instance = AuthorityPolicy()
        logger.info("AuthorityPolicy initialized")
    return _instance
