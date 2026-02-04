"""
Moltbot Server Manager
======================
Manages llama.cpp server process lifecycle, model loading/unloading, VRAM cleanup.
"""

import subprocess
import asyncio
import logging
import signal
import os
import time
from pathlib import Path
from typing import Optional, Literal
from dataclasses import dataclass, field

import psutil
import yaml

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    name: str
    file: str
    context_length: int = 4096
    temperature: float = 0.3
    top_p: float = 0.9
    gpu_layers: int = -1
    offload_experts: bool = False
    expert_offload_pattern: str = ".ffn_.*_exps.=CPU"


@dataclass
class ServerState:
    """Current state of the llama.cpp server."""
    running: bool = False
    model_loaded: Optional[str] = None
    process: Optional[subprocess.Popen] = None
    pid: Optional[int] = None
    start_time: Optional[float] = None


class LlamaServer:
    """Manages llama.cpp server process."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.state = ServerState()
        self.base_dir = self.config_path.parent
        self._health_check_interval = 5.0
        self._shutdown_timeout = 10.0

    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def reload_config(self) -> None:
        """Reload configuration from file."""
        self.config = self._load_config()
        logger.info("Configuration reloaded")

    def _get_model_config(self, model_name: str) -> ModelConfig:
        """Get configuration for a specific model."""
        model_cfg = self.config['models'].get(model_name)
        if not model_cfg:
            raise ValueError(f"Unknown model: {model_name}")
        return ModelConfig(name=model_name, **model_cfg)

    def _get_model_path(self, model_cfg: ModelConfig) -> Path:
        """Get full path to model file."""
        models_dir = self.base_dir / self.config['paths']['models_dir']
        return models_dir / model_cfg.file

    def _build_server_command(self, model_cfg: ModelConfig) -> list[str]:
        """Build llama-server command with all arguments."""
        server_cfg = self.config['server']
        model_path = self._get_model_path(model_cfg)

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Verify llama-server exists
        llama_path = self.base_dir / self.config['paths']['llama_cpp']
        if not llama_path.exists():
            raise FileNotFoundError(f"llama-server not found: {llama_path}")

        # Build path to llama-server (relative to base_dir)
        llama_path = self.base_dir / self.config['paths']['llama_cpp']

        cmd = [
            str(llama_path),
            "--model", str(model_path),
            "--host", server_cfg['host'],
            "--port", str(server_cfg['port']),
            "--ctx-size", str(model_cfg.context_length),
            "--threads", str(server_cfg['threads']),
            "--batch-size", str(server_cfg['batch_size']),
            "--n-gpu-layers", str(model_cfg.gpu_layers),
            "--device", "Vulkan0",  # Use RX 9070 XT (first Vulkan device)
            "--flash-attn", "on",   # Enable flash attention for efficiency
        ]

        # KV cache quantization
        if server_cfg.get('cache_type_k'):
            cmd.extend(["--cache-type-k", server_cfg['cache_type_k']])
        if server_cfg.get('cache_type_v'):
            cmd.extend(["--cache-type-v", server_cfg['cache_type_v']])

        # MoE expert offloading for large models
        if model_cfg.offload_experts:
            cmd.extend(["-ot", model_cfg.expert_offload_pattern])

        return cmd

    async def start(self, model_name: Optional[str] = None) -> bool:
        """
        Start the llama.cpp server with specified model.

        Args:
            model_name: Model to load ('forecaster' or 'coder').
                       Uses default_model from config if not specified.

        Returns:
            True if server started successfully.
        """
        if self.state.running:
            if self.state.model_loaded == model_name:
                logger.info(f"Server already running with {model_name}")
                return True
            # Different model requested - stop first
            await self.stop()

        model_name = model_name or self.config.get('default_model', 'forecaster')
        model_cfg = self._get_model_config(model_name)

        logger.info(f"Starting server with model: {model_cfg.name}")

        try:
            cmd = self._build_server_command(model_cfg)
            logger.debug(f"Server command: {' '.join(cmd)}")

            # Start process
            log_dir = self.base_dir / self.config['paths']['logs_dir']
            log_file = log_dir / f"llama_server_{model_name}.log"

            with open(log_file, 'a') as log_f:
                self.state.process = subprocess.Popen(
                    cmd,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
                )

            self.state.pid = self.state.process.pid
            self.state.start_time = time.time()

            # Wait for server to be ready
            if await self._wait_for_ready():
                self.state.running = True
                self.state.model_loaded = model_name
                logger.info(f"Server started successfully (PID: {self.state.pid})")
                return True
            else:
                logger.error("Server failed to become ready")
                await self.stop()
                return False

        except FileNotFoundError as e:
            logger.error(f"Failed to start server: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error starting server: {e}")
            return False

    async def _wait_for_ready(self, timeout: float = 60.0) -> bool:
        """Wait for server to respond to health check."""
        import aiohttp

        server_cfg = self.config['server']
        health_url = f"http://{server_cfg['host']}:{server_cfg['port']}/health"

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(health_url, timeout=2) as resp:
                        if resp.status == 200:
                            return True
            except Exception:
                pass

            # Check if process died
            if self.state.process and self.state.process.poll() is not None:
                logger.error(f"Server process exited with code: {self.state.process.returncode}")
                return False

            await asyncio.sleep(1.0)

        return False

    async def stop(self) -> bool:
        """
        Stop the llama.cpp server and free VRAM.

        Returns:
            True if server stopped successfully.
        """
        if not self.state.running and not self.state.process:
            logger.info("Server not running")
            return True

        logger.info("Stopping server...")

        try:
            if self.state.process:
                # Try graceful shutdown first
                if os.name == 'nt':
                    self.state.process.terminate()
                else:
                    self.state.process.send_signal(signal.SIGTERM)

                try:
                    self.state.process.wait(timeout=self._shutdown_timeout)
                except subprocess.TimeoutExpired:
                    logger.warning("Graceful shutdown timed out, killing process")
                    self.state.process.kill()
                    self.state.process.wait()

            # Ensure VRAM is freed by killing any orphaned processes
            await self._cleanup_orphans()

            self.state = ServerState()
            logger.info("Server stopped, VRAM freed")
            return True

        except Exception as e:
            logger.error(f"Error stopping server: {e}")
            return False

    async def _cleanup_orphans(self) -> None:
        """Kill any orphaned llama-server processes."""
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if 'llama' in proc.info['name'].lower() and 'server' in proc.info['name'].lower():
                    logger.warning(f"Killing orphaned process: {proc.info['pid']}")
                    proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    async def switch_model(self, model_name: str) -> bool:
        """
        Switch to a different model.

        Args:
            model_name: Model to switch to ('forecaster' or 'coder').

        Returns:
            True if switch was successful.
        """
        if self.state.model_loaded == model_name:
            logger.info(f"Already running {model_name}")
            return True

        logger.info(f"Switching from {self.state.model_loaded} to {model_name}")

        await self.stop()
        return await self.start(model_name)

    def get_status(self) -> dict:
        """Get current server status."""
        status = {
            "running": self.state.running,
            "model": self.state.model_loaded,
            "pid": self.state.pid,
            "uptime": None,
            "vram_usage": None
        }

        if self.state.start_time:
            status["uptime"] = time.time() - self.state.start_time

        # Try to get VRAM usage (AMD-specific)
        try:
            status["vram_usage"] = self._get_vram_usage()
        except Exception:
            pass

        return status

    def _get_vram_usage(self) -> Optional[dict]:
        """Get VRAM usage from AMD GPU."""
        try:
            # Try rocm-smi first
            result = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                # Parse output
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if 'Used' in line:
                        parts = line.split()
                        for i, p in enumerate(parts):
                            if p == 'Used':
                                return {"used_mb": int(parts[i+1])}
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.debug(f"Could not get VRAM usage: {e}")

        return None

    @property
    def api_base(self) -> str:
        """Get the API base URL."""
        server_cfg = self.config['server']
        return f"http://{server_cfg['host']}:{server_cfg['port']}"

    async def health_check(self) -> bool:
        """Check if server is healthy."""
        if not self.state.running:
            return False

        import aiohttp
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_base}/health", timeout=5) as resp:
                    return resp.status == 200
        except Exception:
            return False
