"""
Moltbot Telegram Interface
==========================
Telegram bot that interfaces with the inference engine.
"""

import asyncio
import logging
from typing import Optional, TYPE_CHECKING

from telegram import Update, BotCommand
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters
)
from telegram.constants import ParseMode, ChatAction

if TYPE_CHECKING:
    from core.server import LlamaServer
    from core.inference import InferenceEngine
    from core.state import StateManager

logger = logging.getLogger(__name__)


class MoltbotTelegram:
    """Telegram bot interface for Moltbot."""

    def __init__(
        self,
        token: str,
        allowed_user_ids: list[int],
        server: "LlamaServer",
        inference: "InferenceEngine",
        state_manager: "StateManager"
    ):
        self.token = token
        self.allowed_user_ids = set(allowed_user_ids)
        self.server = server
        self.inference = inference
        self.state_manager = state_manager
        self.app: Optional[Application] = None
        self._running = False

    async def start(self) -> None:
        """Start the Telegram bot."""
        self.app = Application.builder().token(self.token).build()

        # Register handlers
        self.app.add_handler(CommandHandler("start", self._cmd_start))
        self.app.add_handler(CommandHandler("stop", self._cmd_stop))
        self.app.add_handler(CommandHandler("status", self._cmd_status))
        self.app.add_handler(CommandHandler("model", self._cmd_model))
        self.app.add_handler(CommandHandler("clear", self._cmd_clear))
        self.app.add_handler(CommandHandler("help", self._cmd_help))

        # Message handler for inference
        self.app.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            self._handle_message
        ))

        # Set bot commands
        await self.app.bot.set_my_commands([
            BotCommand("start", "Start the inference server"),
            BotCommand("stop", "Stop the server and free VRAM"),
            BotCommand("status", "Show server status"),
            BotCommand("model", "Switch model (forecaster/coder)"),
            BotCommand("clear", "Clear conversation history"),
            BotCommand("help", "Show help message"),
        ])

        # Start polling
        self._running = True
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling(drop_pending_updates=True)
        logger.info("Telegram bot started")

    async def stop(self) -> None:
        """Stop the Telegram bot."""
        if self.app and self._running:
            self._running = False
            await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()
            logger.info("Telegram bot stopped")

    def _is_authorized(self, user_id: int) -> bool:
        """Check if user is authorized."""
        # If no users configured, allow all (for testing)
        if not self.allowed_user_ids or 0 in self.allowed_user_ids:
            return True
        return user_id in self.allowed_user_ids

    async def _unauthorized(self, update: Update) -> None:
        """Send unauthorized message."""
        await update.message.reply_text(
            "You are not authorized to use this bot."
        )

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command - start the inference server."""
        if not self._is_authorized(update.effective_user.id):
            await self._unauthorized(update)
            return

        await update.message.reply_text("Starting inference server...")

        # Get user's current model preference
        conv = self.state_manager.get_conversation(str(update.effective_user.id))
        model = conv.current_model

        success = await self.server.start(model)
        if success:
            await update.message.reply_text(
                f"Server started with **{model}** model.\n"
                f"Send me a message to begin!",
                parse_mode=ParseMode.MARKDOWN
            )
        else:
            await update.message.reply_text(
                "Failed to start server. Check logs for details."
            )

    async def _cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /stop command - stop the server and free VRAM."""
        if not self._is_authorized(update.effective_user.id):
            await self._unauthorized(update)
            return

        await update.message.reply_text("Stopping server...")

        success = await self.server.stop()
        if success:
            await update.message.reply_text(
                "Server stopped. VRAM freed.\n"
                "Use /start to restart."
            )
        else:
            await update.message.reply_text(
                "Error stopping server. Check logs."
            )

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /status command - show server status."""
        if not self._is_authorized(update.effective_user.id):
            await self._unauthorized(update)
            return

        status = self.server.get_status()

        if status['running']:
            uptime_str = ""
            if status['uptime']:
                mins = int(status['uptime'] // 60)
                secs = int(status['uptime'] % 60)
                uptime_str = f"\nUptime: {mins}m {secs}s"

            vram_str = ""
            if status.get('vram_usage'):
                vram_str = f"\nVRAM: {status['vram_usage'].get('used_mb', '?')} MB"

            msg = (
                f"**Status: Running**\n"
                f"Model: {status['model']}\n"
                f"PID: {status['pid']}"
                f"{uptime_str}"
                f"{vram_str}"
            )
        else:
            msg = "**Status: Stopped**\nUse /start to begin."

        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

    async def _cmd_model(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /model command - switch between models."""
        if not self._is_authorized(update.effective_user.id):
            await self._unauthorized(update)
            return

        args = context.args
        if not args:
            # Show current model and options
            conv = self.state_manager.get_conversation(str(update.effective_user.id))
            await update.message.reply_text(
                f"Current model: **{conv.current_model}**\n\n"
                f"Usage: `/model <name>`\n"
                f"Available models:\n"
                f"- `forecaster` - Prediction market analysis\n"
                f"- `coder` - Coding assistance",
                parse_mode=ParseMode.MARKDOWN
            )
            return

        model_name = args[0].lower()
        if model_name not in ['forecaster', 'coder']:
            await update.message.reply_text(
                f"Unknown model: {model_name}\n"
                f"Available: forecaster, coder"
            )
            return

        await update.message.reply_text(f"Switching to **{model_name}**...", parse_mode=ParseMode.MARKDOWN)

        # Update user's preference
        conv = self.state_manager.get_conversation(str(update.effective_user.id))
        conv.current_model = model_name

        # Switch if server is running
        if self.server.state.running:
            success = await self.server.switch_model(model_name)
            if success:
                await update.message.reply_text(f"Switched to **{model_name}**.", parse_mode=ParseMode.MARKDOWN)
            else:
                await update.message.reply_text("Failed to switch model. Check logs.")
        else:
            await update.message.reply_text(
                f"Model set to **{model_name}**.\n"
                f"Will load on next /start.",
                parse_mode=ParseMode.MARKDOWN
            )

    async def _cmd_clear(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /clear command - clear conversation history."""
        if not self._is_authorized(update.effective_user.id):
            await self._unauthorized(update)
            return

        self.state_manager.clear_conversation(str(update.effective_user.id))
        await update.message.reply_text("Conversation history cleared.")

    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command - show help message."""
        help_text = """**Moltbot Commands**

/start - Start the inference server
/stop - Stop server and free GPU memory
/status - Show current status
/model <name> - Switch model
/clear - Clear conversation history
/help - Show this message

**Models:**
- `forecaster` - Prediction market analysis
- `coder` - Coding assistance

**Usage:**
Just send a message to get a response!

For forecaster mode, ask prediction questions like:
"What's the probability that X will happen by Y date?"

For coder mode, ask coding questions or paste code for review.
"""
        await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle regular text messages - run inference."""
        if not self._is_authorized(update.effective_user.id):
            await self._unauthorized(update)
            return

        user_id = str(update.effective_user.id)
        message_text = update.message.text

        # Check if server is running
        if not self.server.state.running:
            await update.message.reply_text(
                "Server is not running.\n"
                "Use /start to begin."
            )
            return

        # Show typing indicator
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action=ChatAction.TYPING
        )

        # Get conversation state
        conv = self.state_manager.get_conversation(user_id)

        try:
            # Run inference
            response = await self.inference.chat(
                user_message=message_text,
                conversation=conv,
                model_name=conv.current_model
            )

            # Send response (split if too long)
            await self._send_long_message(update, response)

        except Exception as e:
            logger.error(f"Inference error: {e}")
            await update.message.reply_text(
                f"Error processing message: {str(e)}"
            )

    async def _send_long_message(self, update: Update, text: str, max_length: int = 4000) -> None:
        """Send a message, splitting if too long for Telegram."""
        if len(text) <= max_length:
            await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)
            return

        # Split on paragraph boundaries
        chunks = []
        current_chunk = ""

        for line in text.split('\n'):
            if len(current_chunk) + len(line) + 1 > max_length:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = line
            else:
                current_chunk += ('\n' if current_chunk else '') + line

        if current_chunk:
            chunks.append(current_chunk)

        for i, chunk in enumerate(chunks):
            try:
                await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)
            except Exception:
                # Fallback without markdown if parsing fails
                await update.message.reply_text(chunk)

            if i < len(chunks) - 1:
                await asyncio.sleep(0.5)  # Rate limit
