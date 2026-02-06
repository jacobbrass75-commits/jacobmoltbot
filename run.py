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
from dotenv import load_dotenv

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
{Fore.CYAN}+==========================================+
|             {Fore.WHITE}M O L T B O T{Fore.CYAN}                |
|      {Fore.YELLOW}OpenAI-Powered Assistant{Fore.CYAN}         |
+==========================================+{Style.RESET_ALL}
"""
    print(banner)


async def main() -> None:
    """Main entry point."""
    print_banner()

    # Determine base directory
    base_dir = Path(__file__).parent
    config_path = base_dir / "config.yaml"

    # Load environment variables from .env (if present)
    env_path = base_dir / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"{Fore.GREEN}Loaded environment from .env{Style.RESET_ALL}")

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
    from core.tools import ToolRegistry
    from core.state import StateManager
    from core.openai_inference import OpenAIInferenceEngine

    # Initialize components
    tool_registry = ToolRegistry()

    # Use OpenAI as the main inference engine
    import os
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print(f"{Fore.RED}Error: OPENAI_API_KEY not set in .env{Style.RESET_ALL}")
        sys.exit(1)

    inference = OpenAIInferenceEngine(tool_registry, model="gpt-4o")
    print(f"{Fore.GREEN}Using OpenAI GPT-4o as primary model{Style.RESET_ALL}")

    data_dir = base_dir / config['paths']['data_dir']
    state_manager = StateManager(
        data_dir=data_dir,
        max_history_tokens=config['state']['max_history_tokens']
    )

    # Initialize External Intelligence Layer (secure multi-AI)
    external_layer = None
    try:
        from core.external import get_external_layer, APIProvider

        log_dir = base_dir / config['paths']['logs_dir']
        external_layer = get_external_layer(log_dir)

        # Report availability (keys never logged)
        availability = external_layer.get_availability()
        print(f"\n{Fore.CYAN}External AI APIs:{Style.RESET_ALL}")
        for api, available in availability.items():
            status = f"{Fore.GREEN}configured" if available else f"{Fore.YELLOW}not configured"
            print(f"  - {api.capitalize()}: {status}{Style.RESET_ALL}")

        # Validate required keys (none required by default - all optional)
        # To require specific APIs, uncomment:
        # success, errors = external_layer.validate_startup([APIProvider.CLAUDE])
        # if not success:
        #     for error in errors:
        #         print(f"{Fore.RED}{error}{Style.RESET_ALL}")
        #     sys.exit(1)

    except ImportError as e:
        print(f"{Fore.YELLOW}External AI layer not available: {e}{Style.RESET_ALL}")
        print("Install with: pip install anthropic openai python-dotenv")

    # Check OpenClaw availability
    try:
        from core.external.openclaw_api import get_openclaw_api
        openclaw = get_openclaw_api()
        if openclaw.is_available():
            print(f"  - OpenClaw: {Fore.GREEN}connected{Style.RESET_ALL}")
        else:
            print(f"  - OpenClaw: {Fore.YELLOW}not running{Style.RESET_ALL}")
    except Exception:
        print(f"  - OpenClaw: {Fore.YELLOW}not available{Style.RESET_ALL}")

    # Initialize ServiceManager
    service_manager = None
    try:
        from core.services import get_service_manager
        service_log_dir = base_dir / config['paths']['logs_dir']
        service_manager = get_service_manager(log_dir=service_log_dir)
        print(f"\n{Fore.CYAN}ServiceManager:{Style.RESET_ALL} {Fore.GREEN}initialized{Style.RESET_ALL}")

        # Auto-start OpenClaw gateway if not already running
        # Note: `openclaw gateway start` is a daemon launcher — it starts the
        # gateway in the background and exits (code 0). So we run it once,
        # not as a restart-on-crash service.
        try:
            import shutil
            openclaw_cli = shutil.which("openclaw")
            if openclaw_cli:
                # Only start if not already connected
                try:
                    from core.external.openclaw_api import get_openclaw_api
                    oc = get_openclaw_api()
                    already_running = oc.is_available()
                except Exception:
                    already_running = False

                if already_running:
                    print(f"  - openclaw-gateway: {Fore.GREEN}already running{Style.RESET_ALL}")
                else:
                    ok, msg = await service_manager.start_service(
                        name="openclaw-gateway",
                        command=f"{openclaw_cli} gateway start",
                        restart_on_crash=False,  # Daemon launcher exits after starting
                    )
                    if ok:
                        print(f"  - openclaw-gateway: {Fore.GREEN}started{Style.RESET_ALL}")
                    else:
                        print(f"  - openclaw-gateway: {Fore.YELLOW}{msg}{Style.RESET_ALL}")
            else:
                print(f"  - openclaw-gateway: {Fore.YELLOW}CLI not found, skipping{Style.RESET_ALL}")
        except Exception as e:
            print(f"  - openclaw-gateway: {Fore.YELLOW}failed to start: {e}{Style.RESET_ALL}")

    except Exception as e:
        print(f"{Fore.YELLOW}ServiceManager not available: {e}{Style.RESET_ALL}")

    # Initialize Authority Policy (logs available actions)
    try:
        from core.authority import get_authority_policy
        get_authority_policy()
        logger.info("AuthorityPolicy loaded")
    except Exception as e:
        logger.warning(f"AuthorityPolicy failed to initialize: {e}")

    # Initialize Telegram bot if configured
    telegram_bot = None
    if telegram_token:
        from tg_interface.openai_bot import OpenAIMoltbot
        telegram_bot = OpenAIMoltbot(
            token=telegram_token,
            allowed_user_ids=config['telegram']['allowed_user_ids'],
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
        if telegram_bot:
            print(f"{Fore.GREEN}Starting Telegram bot...{Style.RESET_ALL}")
            await telegram_bot.start()

        print(f"\n{Fore.GREEN}Moltbot is running!{Style.RESET_ALL}")
        print(f"  - Model: {Fore.CYAN}GPT-4o{Style.RESET_ALL} (OpenAI)")
        if telegram_bot:
            print(f"  - Telegram bot listening")
        print(f"  - Tools: {len(tool_registry.list_tools())} available")
        print(f"\nPress Ctrl+C to quit.\n")

        # Wait for shutdown
        await shutdown_event.wait()

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
    finally:
        # Cleanup
        logger.info("Shutting down...")

        if service_manager:
            await service_manager.shutdown()

        if telegram_bot:
            await telegram_bot.stop()

        state_manager.save_all()

        print(f"\n{Fore.CYAN}Moltbot stopped. Goodbye!{Style.RESET_ALL}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
