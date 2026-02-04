# Moltbot Architecture Documentation

A local LLM inference system with Telegram bot integration and desktop GUI, optimized for AMD Radeon RX 9070 XT with Vulkan backend.

## Overview

Moltbot provides:
- **Local LLM inference** via llama.cpp with Vulkan GPU acceleration
- **Two switchable models**: Forecaster (prediction analysis) and Coder (software engineering)
- **Telegram bot** for remote interaction
- **Desktop GUI** for easy control
- **Tool calling** (web search, calculator, base rates)
- **Conversation persistence** per user

## System Requirements

- **GPU**: AMD Radeon RX 9070 XT (16GB VRAM)
- **RAM**: 32GB DDR5
- **Backend**: Vulkan (auto-detected by llama.cpp)
- **OS**: Windows (tested), Linux (should work)

## Directory Structure

```
moltbot/
├── config.yaml              # User configuration (secrets, settings)
├── config.yaml.example      # Template configuration
├── requirements.txt         # Python dependencies
├── gui.py                   # Desktop GUI with embedded Telegram bot
├── run.py                   # CLI entry point (alternative to GUI)
├── core/                    # Core logic modules
│   ├── __init__.py
│   ├── server.py            # llama.cpp process management
│   ├── inference.py         # Chat completion + tool calling loop
│   ├── tools.py             # Tool implementations (search, calc)
│   ├── prompts.py           # System prompts per model mode
│   └── state.py             # Conversation persistence
├── tg_interface/            # Telegram bot (used by run.py)
│   ├── __init__.py
│   └── bot.py               # Full-featured Telegram interface
├── tray/                    # System tray app (used by run.py)
│   ├── __init__.py
│   └── app.py               # pystray-based tray icon
├── scripts/
│   └── download_models.py   # Model downloader from HuggingFace
├── models/                  # GGUF model files (gitignored)
├── llama-cpp/               # llama.cpp binaries (gitignored)
├── logs/                    # Runtime logs
└── data/
    └── conversations/       # Per-user JSON conversation state
```

## Models

### Forecaster (Qwen2.5-7B-Instruct)
- **Purpose**: Prediction market analysis with structured probability estimates
- **File**: `Qwen2.5-7B-Instruct-Q5_K_M.gguf` (5.44 GB)
- **Quantization**: Q5_K_M (good quality/size balance)
- **VRAM**: ~5.4 GB
- **Context**: 4096 tokens

### Coder (Qwen3-Coder-30B-A3B)
- **Purpose**: Software engineering assistance
- **File**: `Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf` (18.6 GB)
- **Quantization**: Q4_K_M
- **Architecture**: Mixture of Experts (MoE) - only 3B active params
- **VRAM**: ~6-8 GB (with CPU offloading of expert layers)
- **Context**: 8192 tokens
- **MoE Offload**: Expert FFN layers offloaded to CPU with `-ot .ffn_.*_exps.=CPU`

---

## Module Documentation

### `gui.py` - Desktop Control Panel (475 lines)

The primary user interface combining a tkinter GUI with embedded Telegram bot functionality.

#### Classes

**`TelegramBot`** (lines 26-179)
- Lightweight Telegram bot that forwards messages to llama-server
- Maintains per-user conversation history (last 10 messages)
- Commands: `/start`, `/clear`, `/status`
- Handles message chunking for Telegram's 4000 char limit

Key methods:
- `start()` - Initialize and start polling
- `stop()` - Graceful shutdown
- `_handle_message()` - Forward user messages to llama API

**`MoltbotGUI`** (lines 182-474)
- tkinter-based control panel
- Model selection (radio buttons)
- Start/Stop buttons with status indicator
- Log display area
- Telegram enable/disable toggle

Key methods:
- `_start_server()` - Launch llama-server subprocess
- `_start_server_thread()` - Build command, monitor health endpoint
- `_stop_server()` - Terminate process, kill orphans, free VRAM
- `_start_telegram()` - Start bot in separate thread/event loop

---

### `core/server.py` - Server Manager (339 lines)

Manages the llama.cpp server process lifecycle.

#### Data Classes

**`ModelConfig`** - Configuration for a single model
```python
@dataclass
class ModelConfig:
    name: str
    file: str
    context_length: int = 4096
    temperature: float = 0.3
    top_p: float = 0.9
    gpu_layers: int = -1
    offload_experts: bool = False
    expert_offload_pattern: str = ".ffn_.*_exps.=CPU"
```

**`ServerState`** - Current runtime state
```python
@dataclass
class ServerState:
    running: bool = False
    model_loaded: Optional[str] = None
    process: Optional[subprocess.Popen] = None
    pid: Optional[int] = None
    start_time: Optional[float] = None
```

#### Class: `LlamaServer`

Key methods:
- `start(model_name)` - Start server with specified model
- `stop()` - Graceful shutdown, kill process, cleanup orphans
- `switch_model(model_name)` - Stop current, start new model
- `_build_server_command()` - Construct llama-server CLI args
- `_wait_for_ready()` - Poll `/health` endpoint until ready
- `_cleanup_orphans()` - Kill any orphaned llama-server processes
- `health_check()` - Check if server is responding
- `get_status()` - Return status dict (running, model, pid, uptime)

Server command construction (line 79-118):
```bash
llama-server.exe \
  --model <model_path> \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 4096 \
  --threads 8 --batch-size 512 \
  --n-gpu-layers -1 \
  --device Vulkan0 \
  --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q4_0 \
  -ot ".ffn_.*_exps.=CPU"  # MoE offload (coder only)
```

---

### `core/inference.py` - Inference Engine (267 lines)

Handles chat completions with tool calling support.

#### Data Classes

**`Message`** - Single conversation message
```python
@dataclass
class Message:
    role: str  # 'system', 'user', 'assistant', 'tool'
    content: str
    tool_calls: Optional[list[dict]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
```

**`InferenceResult`** - API response wrapper
```python
@dataclass
class InferenceResult:
    content: str
    tool_calls: list[dict]
    finish_reason: str
    tokens_used: int
```

#### Class: `InferenceEngine`

Key methods:
- `chat(user_message, conversation, model_name)` - Main entry point
  - Adds user message to conversation
  - Runs tool calling loop (max 5 rounds)
  - Returns final assistant response
- `_build_messages()` - Construct API payload with history
- `_call_api()` - POST to `/v1/chat/completions`
- `_parse_response()` - Extract content and tool calls
- `_extract_tool_calls_from_content()` - Handle models that embed JSON
- `_execute_tool()` - Run tool and return result

Tool calling loop (lines 81-115):
1. Call API with messages + tool definitions
2. If no tool calls, return response
3. For each tool call:
   - Execute tool
   - Add result to messages
4. Repeat until no more tool calls or max rounds

---

### `core/tools.py` - Tool Registry (272 lines)

Defines and executes tools the model can call.

#### Data Class

**`ToolResult`**
```python
@dataclass
class ToolResult:
    success: bool
    content: str
    error: Optional[str] = None
```

#### Class: `ToolRegistry`

Built-in tools:

**`web_search(query, max_results=5)`** (lines 70-100)
- Uses DuckDuckGo Search API (no API key required)
- Returns formatted search results with title, snippet, URL

**`calculate(expression)`** (lines 102-150)
- Safe math evaluation using `simpleeval`
- Supports: basic arithmetic, sqrt, log, sin, cos, etc.
- Constants: pi, e, inf

**`get_base_rate(event_type, context)`** (lines 152-267)
- Returns historical frequencies for prediction market events
- Categories:
  - Political: incumbent_reelection (~70%), party_retention (~50%), midterm_loss (~90%)
  - Economic: recession_annual (~15%), market_crash_10pct (~30%)
  - Technology: startup_failure (~90%), major_tech_outage (~5-10%)
  - Geopolitical: interstate_war (~1-2%), coup_attempt (~2%)
  - Science/Health: drug_approval (~10%), pandemic_novel (~1%)

---

### `core/prompts.py` - System Prompts (200 lines)

Contains specialized system prompts for each model mode.

#### Forecaster Prompt (lines 7-66)
Instructs the model to:
- Start with base rates before adjusting
- Give probability ranges, not point estimates
- Structure responses with: Event Decomposition, Base Rate, Probability Band, Key Drivers, Uncertainty Notes
- Avoid common biases (availability, confirmation, overconfidence, anchoring)

#### Coder Prompt (lines 68-112)
Instructs the model to:
- Prioritize correctness over cleverness
- Read before write, minimal changes
- Follow code quality standards
- Ask clarifying questions when unsure

#### Functions

**`get_system_prompt(model_name)`** - Returns appropriate prompt

**`get_tool_definitions(model_name)`** - Returns OpenAI function calling format
- Common tools: `web_search`, `calculate`
- Forecaster-only: `get_base_rate`

---

### `core/state.py` - State Management (192 lines)

Handles conversation persistence and context pruning.

#### Data Classes

**`MessageData`** - Serializable message
**`ConversationData`** - Full conversation with metadata

#### Class: `ConversationState`

Manages a single user's conversation:
- `add_message(message)` - Add and auto-save
- `get_messages()` - Return as Message objects
- `clear()` - Clear history
- `_prune_if_needed()` - Remove old messages when exceeding `max_history_tokens`

Storage: `data/conversations/{user_id}.json`

Pruning algorithm (lines 127-139):
- Estimate tokens: `total_chars // 4`
- Remove oldest messages until under limit
- Keep at least 2 messages

#### Class: `StateManager`

Manages all users' conversations:
- `get_conversation(user_id)` - Get or create
- `clear_conversation(user_id)` - Clear specific user
- `save_all()` - Persist all active conversations

---

### `tg_interface/bot.py` - Full Telegram Interface (334 lines)

More feature-rich Telegram bot (used by `run.py`).

#### Class: `MoltbotTelegram`

Commands:
- `/start` - Start inference server
- `/stop` - Stop server, free VRAM
- `/status` - Show current status
- `/model <name>` - Switch model
- `/clear` - Clear conversation history
- `/help` - Show help

Key differences from `gui.py` TelegramBot:
- Uses full `InferenceEngine` with tool calling
- Integrates with `StateManager` for persistence
- Server control via Telegram commands

---

### `tray/app.py` - System Tray (291 lines)

System tray icon with menu controls.

#### Class: `TrayIcon`

- Colored status indicator: Green (running), Yellow (loading), Gray (stopped), Red (error)
- Menu: Start/Stop Server, Switch Model, Open Logs, Quit

#### Class: `TrayApp`

Coordinates tray icon with server:
- `_on_start()` - Start server, update icon
- `_on_stop()` - Stop server, update icon
- `_on_switch_model()` - Switch and update

---

### `scripts/download_models.py` - Model Downloader (205 lines)

Downloads GGUF models from HuggingFace.

Model URLs:
- Forecaster: `huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF`
- Coder: `huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF`

Usage:
```bash
python scripts/download_models.py              # Download all
python scripts/download_models.py forecaster   # Download one
python scripts/download_models.py --yes        # Skip confirmation
python scripts/download_models.py --list       # List available
```

---

## Configuration

### `config.yaml`

```yaml
telegram:
  bot_token: "YOUR_TOKEN"
  allowed_user_ids:
    - 123456789  # Your Telegram user ID

models:
  forecaster:
    name: "Qwen2.5-7B-Instruct"
    file: "Qwen2.5-7B-Instruct-Q5_K_M.gguf"
    context_length: 4096
    temperature: 0.3
    gpu_layers: -1
    offload_experts: false

  coder:
    name: "Qwen3-Coder-30B-A3B"
    file: "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"
    context_length: 8192
    temperature: 0.2
    gpu_layers: -1
    offload_experts: true
    expert_offload_pattern: ".ffn_.*_exps.=CPU"

server:
  host: "127.0.0.1"
  port: 8080
  vulkan: true
  threads: 8
  batch_size: 512
  cache_type_k: "q8_0"
  cache_type_v: "q4_0"

paths:
  models_dir: "models"
  data_dir: "data"
  logs_dir: "logs"
  llama_cpp: "llama-cpp/llama-server.exe"

state:
  save_conversations: true
  max_history_tokens: 2048

default_model: "forecaster"
```

---

## Usage

### Desktop GUI (Recommended)
```bash
python gui.py
# Or double-click the desktop shortcut
```

### CLI Mode
```bash
python run.py
```

### Download Models
```bash
python scripts/download_models.py --yes
```

---

## Extending Moltbot

### Adding a New Model

1. Add model config to `config.yaml`:
```yaml
models:
  new_model:
    name: "Model Name"
    file: "model.gguf"
    context_length: 4096
    temperature: 0.5
    gpu_layers: -1
```

2. Add system prompt to `core/prompts.py`:
```python
NEW_MODEL_PROMPT = """Your system prompt here..."""

def get_system_prompt(model_name):
    prompts = {
        "forecaster": FORECASTER_SYSTEM_PROMPT,
        "coder": CODER_SYSTEM_PROMPT,
        "new_model": NEW_MODEL_PROMPT,  # Add here
    }
```

3. Add tool definitions if needed in `get_tool_definitions()`

### Adding a New Tool

1. Add tool implementation to `core/tools.py`:
```python
def _my_tool(self, param1: str, param2: int = 10) -> str:
    """Tool docstring."""
    # Implementation
    return "result"
```

2. Register in `_register_builtin_tools()`:
```python
self.register("my_tool", self._my_tool)
```

3. Add tool definition to `core/prompts.py`:
```python
{
    "type": "function",
    "function": {
        "name": "my_tool",
        "description": "What the tool does",
        "parameters": {
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "..."},
                "param2": {"type": "integer", "default": 10}
            },
            "required": ["param1"]
        }
    }
}
```

---

## Performance Notes

- **Forecaster**: ~149 tokens/sec prompt processing, ~18.7 tokens/sec generation
- **Coder**: Slower due to MoE CPU offloading, but fits in 16GB VRAM
- **KV Cache**: Quantized (q8_0/q4_0) to save memory
- **Flash Attention**: Enabled for efficiency

---

## Troubleshooting

### Server won't start
- Check `logs/llama_server_*.log` for errors
- Verify model file exists in `models/`
- Ensure llama-cpp binaries are in `llama-cpp/`

### Out of VRAM
- For coder model, ensure `offload_experts: true`
- Reduce `context_length`
- Use smaller quantization (Q3 instead of Q4)

### Telegram bot not responding
- Verify bot token in `config.yaml`
- Check your user ID is in `allowed_user_ids`
- Look for errors in GUI log or `logs/moltbot.log`

### Model loading slowly
- First load compiles shaders (cached after)
- Large models (coder) take longer to load from disk
