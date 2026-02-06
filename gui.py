"""
Moltbot Control Panel - OpenAI Edition
=======================================
GUI for controlling the OpenAI-powered Candace bot.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import asyncio
import threading
import sys
import os
from pathlib import Path
import queue

import yaml
from dotenv import load_dotenv


class MoltbotGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Candace Control Panel")
        self.root.geometry("550x500")
        self.root.resizable(True, True)

        # State
        self.log_queue = queue.Queue()
        self.base_dir = Path(__file__).parent
        self.inference_engine = None
        self.tool_registry = None
        self.state_manager = None
        self.is_running = False

        # Telegram bot
        self.telegram_bot = None
        self.telegram_thread = None
        self.telegram_loop = None

        # Load environment variables from .env
        env_path = self.base_dir / ".env"
        if env_path.exists():
            load_dotenv(env_path)

        # Load config
        self.config = self._load_config()

        # Available OpenAI models
        self.models = {
            "gpt-5.2": "GPT-5.2 (Latest)",
            "gpt-5.2-pro": "GPT-5.2 Pro (Extended reasoning)",
            "gpt-5": "GPT-5",
            "gpt-4o": "GPT-4o",
            "gpt-4o-mini": "GPT-4o Mini (Fast & cheap)",
            "o1": "O1 (Reasoning)",
        }

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

        self.status_label = ttk.Label(status_left, text="Ready", font=("Segoe UI", 12, "bold"))
        self.status_label.pack(side=tk.LEFT)

        self.telegram_status = ttk.Label(status_left, text="", font=("Segoe UI", 9))
        self.telegram_status.pack(side=tk.LEFT, padx=(15, 0))

        self.status_indicator = tk.Canvas(status_frame, width=20, height=20, highlightthickness=0)
        self.status_indicator.pack(side=tk.RIGHT)
        self._update_indicator("gray")

        # Model selection
        model_frame = ttk.LabelFrame(main_frame, text="OpenAI Model", padding="10")
        model_frame.pack(fill=tk.X, pady=(0, 10))

        self.model_var = tk.StringVar(value="gpt-5.2")

        model_row1 = ttk.Frame(model_frame)
        model_row1.pack(fill=tk.X)
        model_row2 = ttk.Frame(model_frame)
        model_row2.pack(fill=tk.X, pady=(5, 0))
        model_row3 = ttk.Frame(model_frame)
        model_row3.pack(fill=tk.X, pady=(5, 0))

        ttk.Radiobutton(model_row1, text="GPT-5.2 (Latest)", value="gpt-5.2", variable=self.model_var).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(model_row1, text="GPT-5.2 Pro", value="gpt-5.2-pro", variable=self.model_var).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(model_row2, text="GPT-5", value="gpt-5", variable=self.model_var).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(model_row2, text="GPT-4o", value="gpt-4o", variable=self.model_var).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(model_row2, text="GPT-4o Mini", value="gpt-4o-mini", variable=self.model_var).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(model_row3, text="O1 (Reasoning)", value="o1", variable=self.model_var).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(model_row3, text="O1 Mini", value="o1-mini", variable=self.model_var).pack(side=tk.LEFT, padx=5)

        # Telegram toggle
        self.telegram_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(model_row1, text="Telegram Bot", variable=self.telegram_var).pack(side=tk.RIGHT)

        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))

        self.start_btn = ttk.Button(button_frame, text="Start Candace", command=self._start, width=18)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.stop_btn = ttk.Button(button_frame, text="Stop", command=self._stop, state=tk.DISABLED, width=12)
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.switch_btn = ttk.Button(button_frame, text="Switch Model", command=self._switch_model, state=tk.DISABLED, width=12)
        self.switch_btn.pack(side=tk.LEFT)

        # Tools info
        tools_frame = ttk.LabelFrame(main_frame, text="Available Tools", padding="10")
        tools_frame.pack(fill=tk.X, pady=(0, 10))

        tools_text = (
            "web_search, calculate, get_base_rate, search_hf_models, download_hf_model,\n"
            "polymarket_search, polymarket_analyze, polymarket_arbitrage, polymarket_positions"
        )
        ttk.Label(tools_frame, text=tools_text, font=("Consolas", 8), foreground="gray").pack(anchor=tk.W)

        # Log area
        log_frame = ttk.LabelFrame(main_frame, text="Log", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, font=("Consolas", 9), state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Bottom info
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(info_frame, text="Backend: OpenAI API | Candace v2.0", font=("Segoe UI", 8)).pack(side=tk.LEFT)

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

    def _start(self):
        model = self.model_var.get()
        self._log(f"Starting Candace with {model}...")
        self._update_indicator("yellow")
        self.status_label.config(text="Starting...")
        self.start_btn.config(state=tk.DISABLED)

        thread = threading.Thread(target=self._start_thread, args=(model,))
        thread.daemon = True
        thread.start()

    def _start_thread(self, model):
        try:
            # Check for OpenAI key
            if not os.getenv("OPENAI_API_KEY"):
                self._log("ERROR: OPENAI_API_KEY not set in .env")
                self.root.after(0, self._on_start_failed)
                return

            # Initialize components
            from core.tools import ToolRegistry
            from core.state import StateManager
            from core.openai_inference import OpenAIInferenceEngine

            self.tool_registry = ToolRegistry()
            self._log(f"Loaded {len(self.tool_registry.list_tools())} tools")

            self.inference_engine = OpenAIInferenceEngine(self.tool_registry, model=model)
            self._log(f"OpenAI engine ready: {model}")

            data_dir = self.base_dir / self.config['paths']['data_dir']
            self.state_manager = StateManager(
                data_dir=data_dir,
                max_history_tokens=self.config['state']['max_history_tokens']
            )

            self.is_running = True
            self.root.after(0, lambda: self._on_start_success(model))

        except Exception as e:
            self._log(f"ERROR: {str(e)}")
            self.root.after(0, self._on_start_failed)

    def _on_start_success(self, model):
        self.status_label.config(text=f"Running ({model})")
        self._update_indicator("#2ecc71")
        self.stop_btn.config(state=tk.NORMAL)
        self.switch_btn.config(state=tk.NORMAL)
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

        def run_telegram():
            self.telegram_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.telegram_loop)
            try:
                from tg_interface.openai_bot import OpenAIMoltbot

                self.telegram_bot = OpenAIMoltbot(
                    token=token,
                    allowed_user_ids=allowed,
                    inference=self.inference_engine,
                    state_manager=self.state_manager
                )

                self.telegram_loop.run_until_complete(self.telegram_bot.start())
                self._log("Telegram bot started!")
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

    def _switch_model(self):
        if not self.inference_engine:
            return

        new_model = self.model_var.get()
        if self.inference_engine.set_model(new_model):
            self._log(f"Switched to {new_model}")
            self.status_label.config(text=f"Running ({new_model})")
        else:
            self._log(f"Failed to switch to {new_model}")

    def _on_start_failed(self):
        self.status_label.config(text="Failed")
        self._update_indicator("#e74c3c")
        self.start_btn.config(state=tk.NORMAL)
        self.is_running = False

    def _stop(self):
        self._log("Stopping...")

        # Stop Telegram first
        self._stop_telegram()

        self.is_running = False
        self.inference_engine = None
        self.tool_registry = None

        self.status_label.config(text="Stopped")
        self._update_indicator("gray")
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.switch_btn.config(state=tk.DISABLED)
        self._log("Stopped.")

    def _on_close(self):
        if self.is_running:
            self._stop()
        self.root.destroy()

    def run(self):
        self._log("Candace Control Panel ready.")
        self._log("Select a model and click 'Start Candace'.")

        # Check for API key
        if os.getenv("OPENAI_API_KEY"):
            self._log("OpenAI API key found.")
        else:
            self._log("WARNING: OPENAI_API_KEY not set in .env")

        self.root.mainloop()


if __name__ == "__main__":
    app = MoltbotGUI()
    app.run()
