"""
Moltbot Desktop Control Panel
=============================
GUI for starting/stopping the server, switching models, and running Telegram bot.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import asyncio
import threading
import subprocess
import time
import sys
import os
from pathlib import Path
import queue
import json

import yaml
import aiohttp
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
from telegram.constants import ChatAction


class TelegramBot:
    """Telegram bot that forwards messages to llama-server."""

    def __init__(self, token: str, allowed_users: list, server_url: str, log_func, model_getter):
        self.token = token
        self.allowed_users = set(allowed_users)
        self.server_url = server_url
        self.log = log_func
        self.get_current_model = model_getter
        self.app = None
        self._running = False
        self.conversations = {}  # user_id -> list of messages

    def _is_authorized(self, user_id: int) -> bool:
        if not self.allowed_users or 0 in self.allowed_users:
            return True
        return user_id in self.allowed_users

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update.effective_user.id):
            await update.message.reply_text("Unauthorized.")
            return
        await update.message.reply_text(
            "Moltbot ready! Send me a message.\n\n"
            "Commands:\n"
            "/clear - Clear conversation\n"
            "/status - Check server status"
        )

    async def _cmd_clear(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update.effective_user.id):
            return
        user_id = update.effective_user.id
        self.conversations[user_id] = []
        await update.message.reply_text("Conversation cleared.")

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update.effective_user.id):
            return
        model = self.get_current_model()
        if model:
            await update.message.reply_text(f"Server running with: {model}")
        else:
            await update.message.reply_text("Server not running.")

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update.effective_user.id):
            await update.message.reply_text("Unauthorized.")
            return

        user_id = update.effective_user.id
        user_message = update.message.text

        # Show typing
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

        # Get or create conversation history
        if user_id not in self.conversations:
            self.conversations[user_id] = []

        # Add user message
        self.conversations[user_id].append({"role": "user", "content": user_message})

        # Keep only last 10 messages to stay within context
        if len(self.conversations[user_id]) > 10:
            self.conversations[user_id] = self.conversations[user_id][-10:]

        # Get current model for system prompt
        model = self.get_current_model()
        if model == "forecaster":
            system_prompt = (
                "You are a prediction market analyst. Provide probability estimates with ranges, "
                "base rates, key drivers, and uncertainty notes. Be conservative and data-driven."
            )
        else:
            system_prompt = (
                "You are an expert software engineer. Provide clear, working code with explanations. "
                "Be concise and focus on correctness."
            )

        messages = [{"role": "system", "content": system_prompt}] + self.conversations[user_id]

        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": 2048,
                    "stream": False
                }
                async with session.post(
                    f"{self.server_url}/v1/chat/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as resp:
                    if resp.status != 200:
                        await update.message.reply_text(f"Server error: {resp.status}")
                        return

                    data = await resp.json()
                    assistant_message = data["choices"][0]["message"]["content"]

                    # Add to history
                    self.conversations[user_id].append({"role": "assistant", "content": assistant_message})

                    # Send response (split if too long)
                    if len(assistant_message) <= 4000:
                        await update.message.reply_text(assistant_message)
                    else:
                        # Split on newlines
                        chunks = []
                        current = ""
                        for line in assistant_message.split('\n'):
                            if len(current) + len(line) + 1 > 4000:
                                chunks.append(current)
                                current = line
                            else:
                                current += ('\n' if current else '') + line
                        if current:
                            chunks.append(current)
                        for chunk in chunks:
                            await update.message.reply_text(chunk)
                            await asyncio.sleep(0.5)

        except asyncio.TimeoutError:
            await update.message.reply_text("Request timed out. Try a shorter message.")
        except aiohttp.ClientError as e:
            await update.message.reply_text(f"Connection error: Is the server running?")
            self.log(f"Telegram error: {e}")
        except Exception as e:
            await update.message.reply_text(f"Error: {str(e)}")
            self.log(f"Telegram error: {e}")

    async def start(self):
        self.app = Application.builder().token(self.token).build()

        self.app.add_handler(CommandHandler("start", self._cmd_start))
        self.app.add_handler(CommandHandler("clear", self._cmd_clear))
        self.app.add_handler(CommandHandler("status", self._cmd_status))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message))

        self._running = True
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling(drop_pending_updates=True)
        self.log("Telegram bot started!")

    async def stop(self):
        if self.app and self._running:
            self._running = False
            await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()
            self.log("Telegram bot stopped.")


class MoltbotGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Moltbot Control Panel")
        self.root.geometry("500x450")
        self.root.resizable(True, True)

        # State
        self.server_process = None
        self.current_model = None
        self.log_queue = queue.Queue()
        self.base_dir = Path(__file__).parent

        # Telegram bot
        self.telegram_bot = None
        self.telegram_thread = None
        self.telegram_loop = None

        # Load config
        self.config = self._load_config()

        # Build UI
        self._build_ui()

        # Start log consumer
        self._consume_logs()

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _load_config(self):
        config_path = self.base_dir / "config.yaml"
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _build_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Status section
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="10")
        status_frame.pack(fill=tk.X, pady=(0, 10))

        status_left = ttk.Frame(status_frame)
        status_left.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.status_label = ttk.Label(status_left, text="Stopped", font=("Segoe UI", 12, "bold"))
        self.status_label.pack(side=tk.LEFT)

        self.telegram_status = ttk.Label(status_left, text="", font=("Segoe UI", 9))
        self.telegram_status.pack(side=tk.LEFT, padx=(15, 0))

        self.status_indicator = tk.Canvas(status_frame, width=20, height=20, highlightthickness=0)
        self.status_indicator.pack(side=tk.RIGHT)
        self._update_indicator("gray")

        # Model selection
        model_frame = ttk.LabelFrame(main_frame, text="Model", padding="10")
        model_frame.pack(fill=tk.X, pady=(0, 10))

        self.model_var = tk.StringVar(value="forecaster")

        models = list(self.config.get('models', {}).keys())
        for model in models:
            rb = ttk.Radiobutton(model_frame, text=model.capitalize(), value=model, variable=self.model_var)
            rb.pack(side=tk.LEFT, padx=10)

        # Telegram toggle
        self.telegram_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(model_frame, text="Enable Telegram", variable=self.telegram_var).pack(side=tk.RIGHT)

        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))

        self.start_btn = ttk.Button(button_frame, text="Start", command=self._start_server, width=15)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.stop_btn = ttk.Button(button_frame, text="Stop", command=self._stop_server, state=tk.DISABLED, width=15)
        self.stop_btn.pack(side=tk.LEFT)

        # Log area
        log_frame = ttk.LabelFrame(main_frame, text="Log", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=12, font=("Consolas", 9), state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Bottom info
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(info_frame, text="GPU: AMD RX 9070 XT | Backend: Vulkan", font=("Segoe UI", 8)).pack(side=tk.LEFT)

    def _update_indicator(self, color):
        self.status_indicator.delete("all")
        self.status_indicator.create_oval(2, 2, 18, 18, fill=color, outline="")

    def _log(self, message):
        self.log_queue.put(message)

    def _consume_logs(self):
        try:
            while True:
                message = self.log_queue.get_nowait()
                self.log_text.config(state=tk.NORMAL)
                self.log_text.insert(tk.END, f"{message}\n")
                self.log_text.see(tk.END)
                self.log_text.config(state=tk.DISABLED)
        except queue.Empty:
            pass
        self.root.after(100, self._consume_logs)

    def _get_current_model(self):
        return self.current_model

    def _start_server(self):
        model = self.model_var.get()
        self._log(f"Starting with {model} model...")
        self._update_indicator("yellow")
        self.status_label.config(text="Starting...")
        self.start_btn.config(state=tk.DISABLED)

        thread = threading.Thread(target=self._start_server_thread, args=(model,))
        thread.daemon = True
        thread.start()

    def _start_server_thread(self, model):
        try:
            model_cfg = self.config['models'].get(model, {})
            server_cfg = self.config['server']

            model_path = self.base_dir / self.config['paths']['models_dir'] / model_cfg['file']
            llama_path = self.base_dir / self.config['paths']['llama_cpp']

            if not model_path.exists():
                self._log(f"ERROR: Model not found: {model_path}")
                self.root.after(0, self._on_start_failed)
                return

            cmd = [
                str(llama_path),
                "--model", str(model_path),
                "--host", server_cfg['host'],
                "--port", str(server_cfg['port']),
                "--ctx-size", str(model_cfg.get('context_length', 4096)),
                "--n-gpu-layers", str(model_cfg.get('gpu_layers', -1)),
                "--device", "Vulkan0",
                "--flash-attn", "on",
                "--threads", str(server_cfg.get('threads', 8)),
            ]

            if model_cfg.get('offload_experts'):
                cmd.extend(["-ot", model_cfg.get('expert_offload_pattern', '.ffn_.*_exps.=CPU')])

            if server_cfg.get('cache_type_k'):
                cmd.extend(["--cache-type-k", server_cfg['cache_type_k']])
            if server_cfg.get('cache_type_v'):
                cmd.extend(["--cache-type-v", server_cfg['cache_type_v']])

            self._log("Starting llama-server...")

            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW
            )

            self.current_model = model

            # Wait for server
            import urllib.request
            health_url = f"http://{server_cfg['host']}:{server_cfg['port']}/health"

            for i in range(90):  # 90 second timeout for larger model
                time.sleep(1)
                if self.server_process.poll() is not None:
                    self._log("ERROR: Server exited unexpectedly")
                    self.root.after(0, self._on_start_failed)
                    return
                try:
                    with urllib.request.urlopen(health_url, timeout=2) as resp:
                        if resp.status == 200:
                            self._log(f"Server ready! Model: {model}")
                            self.root.after(0, lambda: self._on_start_success(model))
                            return
                except:
                    if i % 10 == 0:
                        self._log(f"Loading model... ({i}s)")

            self._log("ERROR: Timeout")
            self.root.after(0, self._on_start_failed)

        except Exception as e:
            self._log(f"ERROR: {str(e)}")
            self.root.after(0, self._on_start_failed)

    def _on_start_success(self, model):
        self.status_label.config(text=f"Running ({model})")
        self._update_indicator("#2ecc71")
        self.stop_btn.config(state=tk.NORMAL)
        self.start_btn.config(state=tk.DISABLED)

        # Start Telegram if enabled
        if self.telegram_var.get():
            self._start_telegram()

    def _start_telegram(self):
        token = self.config['telegram'].get('bot_token', '')
        if not token or token == "YOUR_BOT_TOKEN_HERE":
            self._log("Telegram: No token configured")
            return

        allowed = self.config['telegram'].get('allowed_user_ids', [])
        server_cfg = self.config['server']
        server_url = f"http://{server_cfg['host']}:{server_cfg['port']}"

        self.telegram_bot = TelegramBot(token, allowed, server_url, self._log, self._get_current_model)

        def run_telegram():
            self.telegram_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.telegram_loop)
            try:
                self.telegram_loop.run_until_complete(self.telegram_bot.start())
                self.telegram_loop.run_forever()
            except Exception as e:
                self._log(f"Telegram error: {e}")

        self.telegram_thread = threading.Thread(target=run_telegram, daemon=True)
        self.telegram_thread.start()
        self.telegram_status.config(text="| Telegram: Active")

    def _stop_telegram(self):
        if self.telegram_bot and self.telegram_loop:
            future = asyncio.run_coroutine_threadsafe(self.telegram_bot.stop(), self.telegram_loop)
            try:
                future.result(timeout=5)
            except:
                pass
            self.telegram_loop.call_soon_threadsafe(self.telegram_loop.stop)
            self.telegram_bot = None
            self.telegram_loop = None
        self.telegram_status.config(text="")

    def _on_start_failed(self):
        self.status_label.config(text="Failed")
        self._update_indicator("#e74c3c")
        self.start_btn.config(state=tk.NORMAL)
        self._stop_server()

    def _stop_server(self):
        self._log("Stopping...")

        # Stop Telegram first
        self._stop_telegram()

        if self.server_process:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
            except Exception as e:
                self._log(f"Warning: {e}")
            self.server_process = None

        try:
            subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"],
                          capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
        except:
            pass

        self.current_model = None
        self.status_label.config(text="Stopped")
        self._update_indicator("gray")
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self._log("Stopped. VRAM freed.")

    def _on_close(self):
        if self.server_process:
            self._stop_server()
        self.root.destroy()

    def run(self):
        self._log("Moltbot Control Panel ready.")
        self._log("Select a model and click 'Start'.")
        self.root.mainloop()


if __name__ == "__main__":
    app = MoltbotGUI()
    app.run()
