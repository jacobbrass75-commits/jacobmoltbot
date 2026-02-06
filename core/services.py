"""
Moltbot Service Manager
========================
Manages background processes with crash recovery.
All start/stop operations are authority-checked.
"""

import asyncio
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .authority import get_authority_policy, ActionType

logger = logging.getLogger(__name__)

# Windows subprocess flag
_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0

# Max auto-restarts before disabling crash recovery for a service
_MAX_RESTARTS = 5


@dataclass
class ServiceInfo:
    """Metadata for a managed service."""
    name: str
    command: list[str]
    process: Optional[asyncio.subprocess.Process] = field(default=None, repr=False)
    pid: Optional[int] = None
    running: bool = False
    restart_on_crash: bool = True
    start_time: Optional[float] = None
    restart_count: int = 0
    last_exit_code: Optional[int] = None
    log_file: Optional[Path] = None


class ServiceManager:
    """
    Manages persistent background processes.
    Auto-restarts crashed services up to MAX_RESTARTS times.
    """

    def __init__(self, log_dir: Optional[Path] = None):
        self._services: dict[str, ServiceInfo] = {}
        self._monitor_task: Optional[asyncio.Task] = None
        self._shutdown_flag = False
        self._log_dir = log_dir or Path(__file__).parent.parent / "logs"
        self._log_dir.mkdir(parents=True, exist_ok=True)

    async def start_service(
        self,
        name: str,
        command: str,
        restart_on_crash: bool = True,
    ) -> tuple[bool, str]:
        """
        Start a background service. Authority-checked.

        Args:
            name: Service identifier.
            command: Shell command to run.
            restart_on_crash: Auto-restart if process exits unexpectedly.

        Returns:
            (success, message) tuple.
        """
        # Authority check
        policy = get_authority_policy()
        allowed, reason = policy.is_allowed(ActionType.SERVICE_START)
        if not allowed:
            return False, f"Authority denied: {reason}"

        # Check if already running
        if name in self._services and self._services[name].running:
            return False, f"Service '{name}' is already running (PID {self._services[name].pid})"

        # Parse command
        if sys.platform == "win32":
            cmd_parts = command.split()
        else:
            cmd_parts = command.split()

        # Log file for service output
        log_file = self._log_dir / f"service_{name}.log"

        try:
            log_handle = open(log_file, "a", encoding="utf-8")

            kwargs = {
                "stdout": log_handle,
                "stderr": log_handle,
            }
            if sys.platform == "win32":
                kwargs["creationflags"] = _CREATE_NO_WINDOW

            proc = await asyncio.create_subprocess_exec(*cmd_parts, **kwargs)

            info = ServiceInfo(
                name=name,
                command=cmd_parts,
                process=proc,
                pid=proc.pid,
                running=True,
                restart_on_crash=restart_on_crash,
                start_time=time.monotonic(),
                log_file=log_file,
            )
            self._services[name] = info

            logger.info(f"Service '{name}' started (PID {proc.pid})")

            # Ensure monitor is running
            if self._monitor_task is None or self._monitor_task.done():
                self._monitor_task = asyncio.create_task(self._monitor_loop())

            return True, f"Service '{name}' started (PID {proc.pid})"

        except Exception as e:
            logger.error(f"Failed to start service '{name}': {e}")
            return False, f"Failed to start: {str(e)}"

    async def stop_service(self, name: str) -> tuple[bool, str]:
        """
        Stop a managed service. Authority-checked.

        Args:
            name: Service identifier.

        Returns:
            (success, message) tuple.
        """
        policy = get_authority_policy()
        allowed, reason = policy.is_allowed(ActionType.SERVICE_STOP)
        if not allowed:
            return False, f"Authority denied: {reason}"

        if name not in self._services:
            return False, f"Unknown service: '{name}'"

        info = self._services[name]
        if not info.running or info.process is None:
            return False, f"Service '{name}' is not running"

        # Disable restart so monitor doesn't bring it back
        info.restart_on_crash = False

        try:
            info.process.terminate()
            try:
                await asyncio.wait_for(info.process.wait(), timeout=10)
            except asyncio.TimeoutError:
                info.process.kill()
                await info.process.wait()

            info.running = False
            info.last_exit_code = info.process.returncode
            logger.info(f"Service '{name}' stopped (exit {info.last_exit_code})")
            return True, f"Service '{name}' stopped"

        except Exception as e:
            logger.error(f"Failed to stop service '{name}': {e}")
            return False, f"Failed to stop: {str(e)}"

    async def restart_service(self, name: str) -> tuple[bool, str]:
        """Stop then start a service."""
        if name not in self._services:
            return False, f"Unknown service: '{name}'"

        info = self._services[name]
        cmd_str = " ".join(info.command)
        restart_on_crash = info.restart_on_crash

        if info.running:
            ok, msg = await self.stop_service(name)
            if not ok:
                return False, f"Stop failed: {msg}"

        return await self.start_service(name, cmd_str, restart_on_crash)

    def list_services(self) -> list[dict]:
        """Return status of all managed services."""
        result = []
        for name, info in self._services.items():
            # Check if process is still alive
            if info.process and info.running:
                if info.process.returncode is not None:
                    info.running = False
                    info.last_exit_code = info.process.returncode

            uptime = None
            if info.running and info.start_time:
                uptime = int(time.monotonic() - info.start_time)

            result.append({
                "name": name,
                "running": info.running,
                "pid": info.pid if info.running else None,
                "restart_on_crash": info.restart_on_crash,
                "restart_count": info.restart_count,
                "last_exit_code": info.last_exit_code,
                "uptime_seconds": uptime,
            })
        return result

    async def _monitor_loop(self) -> None:
        """Background loop that checks service health every 5 seconds."""
        logger.info("Service monitor started")

        while not self._shutdown_flag:
            for name, info in list(self._services.items()):
                if not info.running or info.process is None:
                    continue

                # Check if process has exited
                if info.process.returncode is not None:
                    info.running = False
                    info.last_exit_code = info.process.returncode
                    logger.warning(
                        f"Service '{name}' exited (code {info.last_exit_code})"
                    )

                    # Auto-restart if enabled and under limit
                    if info.restart_on_crash and info.restart_count < _MAX_RESTARTS:
                        info.restart_count += 1
                        logger.info(
                            f"Auto-restarting '{name}' "
                            f"(attempt {info.restart_count}/{_MAX_RESTARTS})"
                        )
                        cmd_str = " ".join(info.command)
                        ok, msg = await self.start_service(
                            name, cmd_str, restart_on_crash=True
                        )
                        if ok:
                            # Preserve restart count
                            self._services[name].restart_count = info.restart_count
                        else:
                            logger.error(f"Auto-restart failed for '{name}': {msg}")

                    elif info.restart_count >= _MAX_RESTARTS:
                        info.restart_on_crash = False
                        logger.error(
                            f"Service '{name}' exceeded max restarts ({_MAX_RESTARTS}). "
                            f"Auto-restart disabled."
                        )

            await asyncio.sleep(5)

        logger.info("Service monitor stopped")

    async def shutdown(self) -> None:
        """Stop all services and the monitor loop."""
        self._shutdown_flag = True

        for name in list(self._services.keys()):
            info = self._services[name]
            if info.running:
                # Direct stop without authority check during shutdown
                info.restart_on_crash = False
                if info.process:
                    try:
                        info.process.terminate()
                        await asyncio.wait_for(info.process.wait(), timeout=5)
                    except (asyncio.TimeoutError, ProcessLookupError):
                        try:
                            info.process.kill()
                        except ProcessLookupError:
                            pass
                    info.running = False
                    logger.info(f"Service '{name}' stopped during shutdown")

        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        logger.info("ServiceManager shut down")

    async def register_windows_startup(self, name: str) -> tuple[bool, str]:
        """
        Register a service to start on Windows login via schtasks.

        Args:
            name: Service name (must already be in self._services).

        Returns:
            (success, message) tuple.
        """
        if sys.platform != "win32":
            return False, "Windows startup registration only available on Windows"

        if name not in self._services:
            return False, f"Unknown service: '{name}'"

        info = self._services[name]
        cmd_str = " ".join(info.command)
        task_name = f"Moltbot_{name}"

        # Authority check for shell command
        policy = get_authority_policy()
        schtasks_cmd = f'schtasks /Create /SC ONLOGON /TN "{task_name}" /TR "{cmd_str}" /F'
        allowed, reason = policy.is_shell_command_allowed(schtasks_cmd)
        if not allowed:
            return False, f"Authority denied: {reason}"

        try:
            kwargs = {
                "stdout": asyncio.subprocess.PIPE,
                "stderr": asyncio.subprocess.PIPE,
            }
            if sys.platform == "win32":
                kwargs["creationflags"] = _CREATE_NO_WINDOW

            proc = await asyncio.create_subprocess_exec(
                "schtasks", "/Create", "/SC", "ONLOGON",
                "/TN", task_name, "/TR", cmd_str, "/F",
                **kwargs,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode == 0:
                logger.info(f"Registered '{name}' for Windows startup as '{task_name}'")
                return True, f"Registered '{task_name}' for startup on login"
            else:
                err = stderr.decode("utf-8", errors="replace").strip()
                return False, f"schtasks failed (exit {proc.returncode}): {err}"

        except Exception as e:
            return False, f"Failed: {str(e)}"


# Singleton
_instance: Optional[ServiceManager] = None


def get_service_manager(log_dir: Optional[Path] = None) -> ServiceManager:
    """Get the singleton ServiceManager instance."""
    global _instance
    if _instance is None:
        _instance = ServiceManager(log_dir=log_dir)
        logger.info("ServiceManager initialized")
    return _instance
