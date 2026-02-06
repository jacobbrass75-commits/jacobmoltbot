"""
Moltbot Dependency Resolver
============================
OS-aware package installer. Every install is authority-checked.
"""

import asyncio
import logging
import platform
import shutil
import sys
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from .authority import get_authority_policy, ActionType

logger = logging.getLogger(__name__)

# Windows subprocess flag
_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


class PackageManager(Enum):
    """Supported package managers."""
    NPM = auto()
    PIP = auto()
    WINGET = auto()
    APT = auto()
    BREW = auto()


@dataclass
class InstallResult:
    """Result of a package install attempt."""
    success: bool
    manager: Optional[str] = None
    stdout: str = ""
    stderr: str = ""
    exit_code: Optional[int] = None
    latency_ms: int = 0
    error: Optional[str] = None


class DependencyResolver:
    """
    Resolves and installs missing dependencies.
    All installs route through AuthorityPolicy.
    """

    # Known npm global packages
    _NPM_PACKAGES: set[str] = {
        "openclaw", "piper-tts", "typescript", "ts-node", "nodemon",
        "pm2", "serve", "http-server", "vercel", "netlify-cli",
        "prettier", "eslint", "webpack", "vite",
    }

    # Known pip packages
    _PIP_PACKAGES: set[str] = {
        "flask", "django", "fastapi", "uvicorn", "gunicorn",
        "requests", "httpx", "aiohttp", "beautifulsoup4", "scrapy",
        "numpy", "pandas", "scipy", "matplotlib", "seaborn",
        "torch", "tensorflow", "transformers", "huggingface-hub",
        "openai", "anthropic", "langchain",
        "pytest", "black", "ruff", "mypy", "pylint",
        "pyyaml", "python-dotenv", "colorama", "rich",
        "piper-tts", "TTS", "bark",
    }

    def __init__(self):
        self._available_managers: dict[PackageManager, str] = {}
        self._install_log: list[InstallResult] = []
        self._detect_managers()

    def _detect_managers(self) -> None:
        """Detect which package managers are available on this system."""
        checks = {
            PackageManager.NPM: "npm",
            PackageManager.PIP: "pip",
            PackageManager.WINGET: "winget",
            PackageManager.APT: "apt-get",
            PackageManager.BREW: "brew",
        }
        for mgr, binary in checks.items():
            path = shutil.which(binary)
            if path:
                self._available_managers[mgr] = path
                logger.debug(f"Found package manager: {mgr.name} at {path}")

    def resolve(self, binary_name: str) -> Optional[str]:
        """
        Check if a binary is available on PATH.

        Returns:
            Full path to the binary, or None if not found.
        """
        return shutil.which(binary_name)

    def _auto_detect_manager(self, package: str) -> Optional[PackageManager]:
        """Heuristic: guess which package manager to use for a package."""
        pkg_lower = package.lower().strip()

        if pkg_lower in self._NPM_PACKAGES:
            if PackageManager.NPM in self._available_managers:
                return PackageManager.NPM

        if pkg_lower in self._PIP_PACKAGES:
            if PackageManager.PIP in self._available_managers:
                return PackageManager.PIP

        # Fallback: if only one manager is available, use it
        if len(self._available_managers) == 1:
            return next(iter(self._available_managers))

        # On Windows, prefer winget for unknown packages
        if sys.platform == "win32" and PackageManager.WINGET in self._available_managers:
            return PackageManager.WINGET

        return None

    def _build_install_command(self, mgr: PackageManager, package: str) -> list[str]:
        """Build the install command for a given manager and package."""
        binary = self._available_managers[mgr]

        if mgr == PackageManager.NPM:
            return [binary, "install", "-g", package]
        elif mgr == PackageManager.PIP:
            return [binary, "install", package]
        elif mgr == PackageManager.WINGET:
            return [binary, "install", "--accept-package-agreements", "--accept-source-agreements", package]
        elif mgr == PackageManager.APT:
            return [binary, "install", "-y", package]
        elif mgr == PackageManager.BREW:
            return [binary, "install", package]
        else:
            return [binary, "install", package]

    async def install(
        self,
        package: str,
        manager: str = "auto",
    ) -> InstallResult:
        """
        Install a package. Authority-checked.

        Args:
            package: Package name to install.
            manager: "auto", "npm", "pip", "winget", "apt", "brew".

        Returns:
            InstallResult with success status and output.
        """
        # Authority check
        policy = get_authority_policy()
        allowed, reason = policy.is_allowed(ActionType.INSTALL_DEPENDENCY)
        if not allowed:
            result = InstallResult(success=False, error=f"Authority denied: {reason}")
            self._install_log.append(result)
            return result

        # Resolve manager
        if manager == "auto":
            mgr = self._auto_detect_manager(package)
            if mgr is None:
                result = InstallResult(
                    success=False,
                    error=f"Cannot auto-detect package manager for '{package}'. "
                          f"Available managers: {[m.name for m in self._available_managers]}"
                )
                self._install_log.append(result)
                return result
        else:
            try:
                mgr = PackageManager[manager.upper()]
            except KeyError:
                result = InstallResult(
                    success=False,
                    error=f"Unknown manager: {manager}. Use: npm, pip, winget, apt, brew"
                )
                self._install_log.append(result)
                return result

            if mgr not in self._available_managers:
                result = InstallResult(
                    success=False,
                    error=f"{mgr.name} is not available on this system"
                )
                self._install_log.append(result)
                return result

        # Build command
        cmd = self._build_install_command(mgr, package)
        logger.info(f"Installing {package} via {mgr.name}: {' '.join(cmd)}")

        start = time.monotonic()
        try:
            kwargs = {
                "stdout": asyncio.subprocess.PIPE,
                "stderr": asyncio.subprocess.PIPE,
            }
            if sys.platform == "win32":
                kwargs["creationflags"] = _CREATE_NO_WINDOW

            proc = await asyncio.create_subprocess_exec(*cmd, **kwargs)

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=300  # 5 minutes
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                result = InstallResult(
                    success=False,
                    manager=mgr.name,
                    error=f"Install timed out after 5 minutes",
                    latency_ms=int((time.monotonic() - start) * 1000),
                )
                self._install_log.append(result)
                return result

            elapsed = int((time.monotonic() - start) * 1000)
            stdout_text = stdout_bytes.decode("utf-8", errors="replace").strip()
            stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()

            success = proc.returncode == 0
            result = InstallResult(
                success=success,
                manager=mgr.name,
                stdout=stdout_text,
                stderr=stderr_text,
                exit_code=proc.returncode,
                latency_ms=elapsed,
                error=None if success else f"Exit code {proc.returncode}",
            )

            if success:
                logger.info(f"Installed {package} via {mgr.name} in {elapsed}ms")
            else:
                logger.error(f"Failed to install {package}: exit {proc.returncode}")

            self._install_log.append(result)
            return result

        except Exception as e:
            elapsed = int((time.monotonic() - start) * 1000)
            result = InstallResult(
                success=False,
                manager=mgr.name,
                error=str(e),
                latency_ms=elapsed,
            )
            self._install_log.append(result)
            logger.error(f"Install exception for {package}: {e}")
            return result

    def get_install_log(self) -> list[InstallResult]:
        """Return the full install log."""
        return list(self._install_log)

    def get_available_managers(self) -> list[str]:
        """List available package managers."""
        return [m.name for m in self._available_managers]


# Singleton
_instance: Optional[DependencyResolver] = None


def get_dependency_resolver() -> DependencyResolver:
    """Get the singleton DependencyResolver instance."""
    global _instance
    if _instance is None:
        _instance = DependencyResolver()
        logger.info(
            f"DependencyResolver initialized. Managers: "
            f"{[m.name for m in _instance._available_managers]}"
        )
    return _instance
