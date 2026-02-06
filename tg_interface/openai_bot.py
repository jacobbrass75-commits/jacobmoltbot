"""
Moltbot OpenAI Telegram Bot
===========================
Simplified Telegram bot using OpenAI API.
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
    from core.openai_inference import OpenAIInferenceEngine
    from core.state import StateManager

logger = logging.getLogger(__name__)


class OpenAIMoltbot:
    """OpenAI-powered Telegram bot."""

    def __init__(
        self,
        token: str,
        allowed_user_ids: list[int],
        inference: "OpenAIInferenceEngine",
        state_manager: "StateManager"
    ):
        self.token = token
        self.allowed_user_ids = set(allowed_user_ids)
        self.inference = inference
        self.state_manager = state_manager
        self.app: Optional[Application] = None
        self._running = False

    async def start(self) -> None:
        """Start the Telegram bot."""
        self.app = Application.builder().token(self.token).build()

        # Register handlers
        self.app.add_handler(CommandHandler("start", self._cmd_start))
        self.app.add_handler(CommandHandler("model", self._cmd_model))
        self.app.add_handler(CommandHandler("clear", self._cmd_clear))
        self.app.add_handler(CommandHandler("status", self._cmd_status))
        self.app.add_handler(CommandHandler("help", self._cmd_help))

        # Message handler for inference
        self.app.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            self._handle_message
        ))

        # Set bot commands
        await self.app.bot.set_my_commands([
            BotCommand("start", "Start chatting"),
            BotCommand("model", "Switch model (gpt-4o, gpt-4o-mini, o1)"),
            BotCommand("clear", "Clear conversation history"),
            BotCommand("status", "Show current status"),
            BotCommand("help", "Show help message"),
        ])

        # Start polling
        self._running = True
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling(drop_pending_updates=True)
        logger.info("OpenAI Telegram bot started")

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
        if not self.allowed_user_ids or 0 in self.allowed_user_ids:
            return True
        return user_id in self.allowed_user_ids

    async def _unauthorized(self, update: Update) -> None:
        """Send unauthorized message."""
        await update.message.reply_text("You are not authorized to use this bot.")

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command."""
        if not self._is_authorized(update.effective_user.id):
            await self._unauthorized(update)
            return

        status = self.inference.get_status()
        await update.message.reply_text(
            f"Hi! I'm Candace, powered by **{status['model']}**.\n\n"
            f"Send me a message to chat!\n\n"
            f"Commands:\n"
            f"/model - Switch models\n"
            f"/clear - Clear history\n"
            f"/status - Current status\n"
            f"/help - More info",
            parse_mode=ParseMode.MARKDOWN
        )

    async def _cmd_model(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /model command - switch models."""
        if not self._is_authorized(update.effective_user.id):
            await self._unauthorized(update)
            return

        args = context.args
        if not args:
            status = self.inference.get_status()
            models = status.get('available_models', [])
            await update.message.reply_text(
                f"Current model: **{status['model']}**\n\n"
                f"Available models:\n"
                f"- `gpt-4o` - Most capable, best for complex tasks\n"
                f"- `gpt-4o-mini` - Fast and cheap\n"
                f"- `o1` - Advanced reasoning\n"
                f"- `o1-mini` - Fast reasoning\n\n"
                f"Usage: `/model gpt-4o-mini`",
                parse_mode=ParseMode.MARKDOWN
            )
            return

        model_name = args[0].lower()
        if self.inference.set_model(model_name):
            await update.message.reply_text(
                f"Switched to **{model_name}**",
                parse_mode=ParseMode.MARKDOWN
            )
        else:
            await update.message.reply_text(f"Unknown model: {model_name}")

    async def _cmd_clear(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /clear command."""
        if not self._is_authorized(update.effective_user.id):
            await self._unauthorized(update)
            return

        self.state_manager.clear_conversation(str(update.effective_user.id))
        await update.message.reply_text("Conversation history cleared.")

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /status command."""
        if not self._is_authorized(update.effective_user.id):
            await self._unauthorized(update)
            return

        status = self.inference.get_status()
        tools = self.inference.tools.list_tools()

        await update.message.reply_text(
            f"**Status**\n"
            f"- Model: {status['model']}\n"
            f"- Backend: {status['backend']}\n"
            f"- Tools: {len(tools)} available\n\n"
            f"**Tools:** {', '.join(tools[:10])}{'...' if len(tools) > 10 else ''}",
            parse_mode=ParseMode.MARKDOWN
        )

    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command."""
        await update.message.reply_text(
            "**Candace - OpenAI-Powered Assistant**\n\n"
            "I can help with:\n"
            "- General questions and conversation\n"
            "- Web search (current information)\n"
            "- Calculations and math\n"
            "- Polymarket analysis (search, arbitrage)\n"
            "- HuggingFace model search\n"
            "- Prediction market forecasting\n\n"
            "**Commands:**\n"
            "/model - Switch AI models\n"
            "/clear - Clear conversation\n"
            "/status - Show current status\n\n"
            "Just send a message to chat!",
            parse_mode=ParseMode.MARKDOWN
        )

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle regular text messages - run inference."""
        if not self._is_authorized(update.effective_user.id):
            await self._unauthorized(update)
            return

        user_id = str(update.effective_user.id)
        message_text = update.message.text

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
                conversation=conv
            )

            # Send response (split if too long)
            await self._send_long_message(update, response)

        except Exception as e:
            logger.error(f"Inference error: {e}")
            await update.message.reply_text(f"Error: {str(e)}")

    async def _send_long_message(self, update: Update, text: str, max_length: int = 4000) -> None:
        """Send a message, splitting if too long for Telegram."""
        if len(text) <= max_length:
            try:
                await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)
            except Exception:
                await update.message.reply_text(text)
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
                await update.message.reply_text(chunk)

            if i < len(chunks) - 1:
                await asyncio.sleep(0.5)
