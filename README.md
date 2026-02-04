# Moltbot

Local LLM inference system with Telegram interface and system tray control.

## Features

- **Two model modes:**
  - `forecaster` - Qwen2.5-7B for prediction market analysis
  - `coder` - Qwen3-Coder-30B-A3B for coding tasks
- **Telegram bot interface** - Chat with your models via Telegram
- **System tray control** - Start/stop/switch models from tray icon
- **Tool calling** - Web search, calculator, base rate lookup
- **State persistence** - Conversations saved and restored
- **VRAM management** - Clean unload when stopped

## Requirements

- Windows 10/11
- Python 3.10+
- AMD RX 9070 XT (16GB VRAM) or similar
- 32GB RAM
- ~25GB disk space for models
- llama.cpp with Vulkan support

## Quick Start

1. **Run setup:**
   ```
   cd moltbot
   scripts\setup.bat
   ```

2. **Configure Telegram:**
   - Talk to [@BotFather](https://t.me/BotFather) on Telegram
   - Create a bot with `/newbot`
   - Copy the token to `config.yaml`
   - Get your user ID from [@userinfobot](https://t.me/userinfobot)
   - Add your ID to `allowed_user_ids` in `config.yaml`

3. **Start Moltbot:**
   ```
   start.bat
   ```

## Telegram Commands

| Command | Description |
|---------|-------------|
| `/start` | Start the inference server |
| `/stop` | Stop server and free VRAM |
| `/status` | Show current status |
| `/model forecaster` | Switch to prediction mode |
| `/model coder` | Switch to coding mode |
| `/clear` | Clear conversation history |
| `/help` | Show help |

## System Tray

Right-click the tray icon for:
- Start/Stop server
- Switch models
- View status
- Open logs
- Quit

## Configuration

Edit `config.yaml` to adjust:

```yaml
# Context length (tokens)
models:
  forecaster:
    context_length: 4096  # Increase for longer conversations
  coder:
    context_length: 8192

# Temperature (0.0-1.0, lower = more deterministic)
models:
  forecaster:
    temperature: 0.3
  coder:
    temperature: 0.2

# Quantization - to switch Q4/Q5, download different GGUF and update filename
models:
  forecaster:
    file: "qwen2.5-7b-instruct-q5_k_m.gguf"  # or q4_k_m
```

## Adding a Second Model Later

1. Download the GGUF to `models/`
2. Add config section in `config.yaml`:
   ```yaml
   models:
     critic:  # New model
       name: "Model Name"
       file: "model-file.gguf"
       context_length: 4096
       temperature: 0.5
   ```
3. Add system prompt in `core/prompts.py`
4. Restart Moltbot

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Telegram Bot   │────▶│  Inference Core  │────▶│ llama.cpp server│
│  (interface)    │◀────│  (tools + logic) │◀────│ (Vulkan/Qwen)   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
        │                        │
        ▼                        ▼
┌─────────────────┐     ┌──────────────────┐
│  System Tray    │     │  State Manager   │
│  (GUI control)  │     │  (persistence)   │
└─────────────────┘     └──────────────────┘
```

## Troubleshooting

### Server won't start
- Check `logs/llama_server_*.log` for errors
- Verify llama-server.exe is in PATH
- Ensure model files exist in `models/`

### Out of memory
- Reduce `context_length` in config
- Use Q4 quantization instead of Q5
- Close other GPU applications

### Telegram bot not responding
- Verify bot token in config
- Check your user ID is in `allowed_user_ids`
- Look at `logs/moltbot.log`

### Slow inference
- MoE offload to RAM is working correctly
- Speed depends on RAM bandwidth
- Expected: 5-15 tokens/sec for coder model

## Files

```
moltbot/
├── config.yaml          # Configuration
├── run.py               # Main entry point
├── start.bat            # Quick start script
├── core/
│   ├── server.py        # llama.cpp management
│   ├── inference.py     # Chat + tool loop
│   ├── tools.py         # Web search, calc, base rates
│   ├── prompts.py       # System prompts
│   └── state.py         # Conversation persistence
├── telegram/
│   └── bot.py           # Telegram interface
├── tray/
│   └── app.py           # System tray icon
├── models/              # GGUF model files
├── data/conversations/  # Saved conversations
└── logs/                # Log files
```
