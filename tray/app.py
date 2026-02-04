"""
Moltbot System Tray Application
===============================
System tray icon with start/stop/status controls.
"""

import asyncio
import logging
import threading
from typing import Optional, Callable, TYPE_CHECKING
from pathlib import Path
import io

from PIL import Image, ImageDraw
import pystray
from pystray import MenuItem, Menu

if TYPE_CHECKING:
    from core.server import LlamaServer

logger = logging.getLogger(__name__)


class TrayIcon:
    """System tray icon for Moltbot."""

    # Icon colors
    COLOR_RUNNING = (46, 204, 113)    # Green
    COLOR_LOADING = (241, 196, 15)    # Yellow
    COLOR_STOPPED = (149, 165, 166)   # Gray
    COLOR_ERROR = (231, 76, 60)       # Red

    def __init__(
        self,
        server: "LlamaServer",
        on_start: Callable,
        on_stop: Callable,
        on_switch_model: Callable,
        on_quit: Callable
    ):
        self.server = server
        self.on_start = on_start
        self.on_stop = on_stop
        self.on_switch_model = on_switch_model
        self.on_quit = on_quit

        self._icon: Optional[pystray.Icon] = None
        self._current_state = "stopped"
        self._thread: Optional[threading.Thread] = None

    def _create_icon_image(self, color: tuple) -> Image.Image:
        """Create a simple colored circle icon."""
        size = 64
        image = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)

        # Draw filled circle
        margin = 4
        draw.ellipse(
            [margin, margin, size - margin, size - margin],
            fill=color,
            outline=(255, 255, 255, 200),
            width=2
        )

        # Add "M" letter in center
        try:
            from PIL import ImageFont
            # Try to use a system font
            font = ImageFont.truetype("arial.ttf", 28)
        except Exception:
            font = ImageFont.load_default()

        text = "M"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (size - text_width) // 2
        y = (size - text_height) // 2 - 2
        draw.text((x, y), text, fill=(255, 255, 255), font=font)

        return image

    def _get_icon_for_state(self, state: str) -> Image.Image:
        """Get icon image for current state."""
        colors = {
            "running": self.COLOR_RUNNING,
            "loading": self.COLOR_LOADING,
            "stopped": self.COLOR_STOPPED,
            "error": self.COLOR_ERROR
        }
        return self._create_icon_image(colors.get(state, self.COLOR_STOPPED))

    def _get_status_text(self) -> str:
        """Get status text for tooltip."""
        status = self.server.get_status()
        if status['running']:
            return f"Moltbot - Running ({status['model']})"
        return "Moltbot - Stopped"

    def _build_menu(self) -> Menu:
        """Build the tray menu."""
        status = self.server.get_status()
        is_running = status['running']
        current_model = status.get('model', 'forecaster')

        return Menu(
            MenuItem(
                "Status: " + ("Running" if is_running else "Stopped"),
                None,
                enabled=False
            ),
            MenuItem(
                f"Model: {current_model}" if is_running else "Model: -",
                None,
                enabled=False
            ),
            Menu.SEPARATOR,
            MenuItem(
                "Start Server",
                self._menu_start,
                enabled=not is_running
            ),
            MenuItem(
                "Stop Server",
                self._menu_stop,
                enabled=is_running
            ),
            Menu.SEPARATOR,
            MenuItem(
                "Switch Model",
                Menu(
                    MenuItem(
                        "Forecaster",
                        lambda: self._menu_switch_model("forecaster"),
                        checked=lambda item: current_model == "forecaster",
                        radio=True
                    ),
                    MenuItem(
                        "Coder",
                        lambda: self._menu_switch_model("coder"),
                        checked=lambda item: current_model == "coder",
                        radio=True
                    ),
                ),
                enabled=is_running
            ),
            Menu.SEPARATOR,
            MenuItem(
                "Open Logs",
                self._menu_open_logs
            ),
            Menu.SEPARATOR,
            MenuItem(
                "Quit",
                self._menu_quit
            ),
        )

    def _menu_start(self, icon, item) -> None:
        """Handle Start menu item."""
        self._run_async(self.on_start)

    def _menu_stop(self, icon, item) -> None:
        """Handle Stop menu item."""
        self._run_async(self.on_stop)

    def _menu_switch_model(self, model: str) -> None:
        """Handle Switch Model menu item."""
        self._run_async(lambda: self.on_switch_model(model))

    def _menu_open_logs(self, icon, item) -> None:
        """Open logs directory."""
        import os
        import subprocess
        logs_dir = Path(self.server.base_dir) / "logs"
        logs_dir.mkdir(exist_ok=True)
        if os.name == 'nt':
            subprocess.run(['explorer', str(logs_dir)])
        else:
            subprocess.run(['xdg-open', str(logs_dir)])

    def _menu_quit(self, icon, item) -> None:
        """Handle Quit menu item."""
        self._run_async(self.on_quit)
        self._icon.stop()

    def _run_async(self, coro_or_func) -> None:
        """Run an async function from sync context."""
        if asyncio.iscoroutinefunction(coro_or_func):
            # Get or create event loop
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            asyncio.run_coroutine_threadsafe(coro_or_func(), loop)
        else:
            # Regular function, just call it
            threading.Thread(target=coro_or_func, daemon=True).start()

    def update_state(self, state: str) -> None:
        """Update the tray icon state."""
        self._current_state = state
        if self._icon:
            self._icon.icon = self._get_icon_for_state(state)
            self._icon.title = self._get_status_text()
            # Rebuild menu to reflect new state
            self._icon.menu = self._build_menu()

    def start(self) -> None:
        """Start the tray icon in a separate thread."""
        def run_tray():
            self._icon = pystray.Icon(
                name="moltbot",
                icon=self._get_icon_for_state(self._current_state),
                title=self._get_status_text(),
                menu=self._build_menu()
            )
            self._icon.run()

        self._thread = threading.Thread(target=run_tray, daemon=True)
        self._thread.start()
        logger.info("System tray icon started")

    def stop(self) -> None:
        """Stop the tray icon."""
        if self._icon:
            self._icon.stop()
            logger.info("System tray icon stopped")


class TrayApp:
    """Main tray application that coordinates components."""

    def __init__(self, server: "LlamaServer", loop: asyncio.AbstractEventLoop):
        self.server = server
        self.loop = loop
        self.tray: Optional[TrayIcon] = None

    async def _on_start(self) -> None:
        """Called when Start is clicked."""
        self.tray.update_state("loading")
        success = await self.server.start()
        self.tray.update_state("running" if success else "error")

    async def _on_stop(self) -> None:
        """Called when Stop is clicked."""
        self.tray.update_state("loading")
        await self.server.stop()
        self.tray.update_state("stopped")

    async def _on_switch_model(self, model: str) -> None:
        """Called when model is switched."""
        self.tray.update_state("loading")
        success = await self.server.switch_model(model)
        self.tray.update_state("running" if success else "error")

    async def _on_quit(self) -> None:
        """Called when Quit is clicked."""
        await self.server.stop()

    def start(self) -> None:
        """Start the tray application."""
        self.tray = TrayIcon(
            server=self.server,
            on_start=lambda: asyncio.run_coroutine_threadsafe(self._on_start(), self.loop),
            on_stop=lambda: asyncio.run_coroutine_threadsafe(self._on_stop(), self.loop),
            on_switch_model=lambda m: asyncio.run_coroutine_threadsafe(self._on_switch_model(m), self.loop),
            on_quit=lambda: asyncio.run_coroutine_threadsafe(self._on_quit(), self.loop)
        )

        # Set initial state
        if self.server.state.running:
            self.tray.update_state("running")
        else:
            self.tray.update_state("stopped")

        self.tray.start()

    def stop(self) -> None:
        """Stop the tray application."""
        if self.tray:
            self.tray.stop()

    def update_state(self, state: str) -> None:
        """Update tray state from outside."""
        if self.tray:
            self.tray.update_state(state)
