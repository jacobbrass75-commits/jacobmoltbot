"""
Moltbot - Main Entry Point
==========================
Orchestrates the tray app, Telegram bot, and inference server.
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path

import yaml
from colorama import init as colorama_init, Fore, Style

# Initialize colorama for Windows
colorama_init()

# Setup logging
def setup_logging(log_dir: Path) -> None:
    """Configure logging to file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "moltbot.log"

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Reduce noise from libraries
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('telegram').setLevel(logging.WARNING)


def print_banner() -> None:
    """Print startup banner."""
    banner = f"""
{Fore.CYAN}╔══════════════════════════════════════════╗
║             {Fore.WHITE}M O L T B O T{Fore.CYAN}                ║
║      {Fore.YELLOW}Local LLM Inference System{Fore.CYAN}         ║
╚══════════════════════════════════════════╝{Style.RESET_ALL}
"""
    print(banner)


async def main() -> None:
    """Main entry point."""
    print_banner()

    # Determine base directory
    base_dir = Path(__file__).parent
    config_path = base_dir / "config.yaml"

    # Load config
    if not config_path.exists():
        print(f"{Fore.RED}Error: config.yaml not found at {config_path}{Style.RESET_ALL}")
        print("Please copy config.yaml.example to config.yaml and configure it.")
        sys.exit(1)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup logging
    log_dir = base_dir / config['paths']['logs_dir']
    setup_logging(log_dir)

    logger = logging.getLogger(__name__)
    logger.info("Moltbot starting...")

    # Check for Telegram token
    telegram_token = config['telegram']['bot_token']
    if telegram_token == "YOUR_BOT_TOKEN_HERE":
        print(f"{Fore.YELLOW}Warning: Telegram bot token not configured.{Style.RESET_ALL}")
        print("Edit config.yaml and set your bot token from @BotFather.")
        print("Telegram bot will not start.\n")
        telegram_token = None

    # Import components
    from core.server import LlamaServer
    from core.inference import InferenceEngine
    from core.tools import ToolRegistry
    from core.state import StateManager
    from tray.app import TrayApp

    # Initialize components
    server = LlamaServer(config_path)
    tool_registry = ToolRegistry()
    inference = InferenceEngine(server, tool_registry)

    data_dir = base_dir / config['paths']['data_dir']
    state_manager = StateManager(
        data_dir=data_dir,
        max_history_tokens=config['state']['max_history_tokens']
    )

    # Get event loop
    loop = asyncio.get_running_loop()

    # Initialize tray app
    tray_app = TrayApp(server, loop)

    # Initialize Telegram bot if configured
    telegram_bot = None
    if telegram_token:
        from telegram.bot import MoltbotTelegram
        telegram_bot = MoltbotTelegram(
            token=telegram_token,
            allowed_user_ids=config['telegram']['allowed_user_ids'],
            server=server,
            inference=inference,
            state_manager=state_manager
        )

    # Setup shutdown handler
    shutdown_event = asyncio.Event()

    def handle_shutdown(signum, frame):
        logger.info("Shutdown signal received")
        shutdown_event.set()

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    try:
        # Start components
        print(f"{Fore.GREEN}Starting system tray...{Style.RESET_ALL}")
        tray_app.start()

        if telegram_bot:
            print(f"{Fore.GREEN}Starting Telegram bot...{Style.RESET_ALL}")
            await telegram_bot.start()

        print(f"\n{Fore.GREEN}Moltbot is running!{Style.RESET_ALL}")
        print(f"  - System tray icon active")
        if telegram_bot:
            print(f"  - Telegram bot listening")
        print(f"  - Server: {Fore.YELLOW}Not started{Style.RESET_ALL} (use tray or /start)")
        print(f"\nPress Ctrl+C to quit.\n")

        # Wait for shutdown
        await shutdown_event.wait()

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
    finally:
        # Cleanup
        logger.info("Shutting down...")

        if telegram_bot:
            await telegram_bot.stop()

        await server.stop()
        tray_app.stop()
        state_manager.save_all()

        print(f"\n{Fore.CYAN}Moltbot stopped. Goodbye!{Style.RESET_ALL}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
