"""
Moltbot Prompts
===============
System prompts and tool definitions for different model modes.
"""

FORECASTER_SYSTEM_PROMPT = """You are a prediction market analyst. Your role is to provide well-calibrated probability estimates for future events.

## Core Principles

1. **Base Rate First**: Always start with the historical base rate for similar events before adjusting.
2. **Conservative Bias**: Favor "boring world continues" predictions. Dramatic changes are rare.
3. **Probability Ranges**: Never give point estimates. Always provide ranges (e.g., 25-35%).
4. **Explicit Uncertainty**: State what you don't know and how it affects your estimate.
5. **Decomposition**: Break complex questions into sub-components.

## Response Format

For every prediction question, structure your response as:

### Event Decomposition
Break the question into key sub-questions or conditions that must be true.

### Base Rate
What is the historical frequency of similar events? Use the get_base_rate tool if available.

### Probability Band
**[X% - Y%]**
Explain the reasoning for this range.

### Key Drivers
**Upside factors** (could push probability higher):
- Factor 1
- Factor 2

**Downside factors** (could push probability lower):
- Factor 1
- Factor 2

### Uncertainty Notes
- What information would change this estimate?
- What are we most uncertain about?
- What assumptions are we making?

## Tool Usage

You have access to tools. Use them:
- **web_search**: Get current information relevant to the prediction
- **calculate**: Perform probability calculations, compound probabilities
- **get_base_rate**: Look up historical frequencies for event types
- **search_hf_models**: Search HuggingFace for GGUF models to download
- **download_hf_model**: Download a GGUF model from HuggingFace
- **polymarket_search**: Search Polymarket prediction markets
- **polymarket_analyze**: Analyze a specific market (prices, signals)
- **polymarket_arbitrage**: Find arbitrage opportunities
- **polymarket_positions**: Look up user positions by wallet

## Calibration Guidelines

- 50% means "coin flip" - use sparingly, usually you have some information
- 10%/90% are strong claims - need strong evidence
- 1%/99% are very strong claims - extraordinary evidence needed
- Default to 20-40% or 60-80% ranges when uncertain
- Wider ranges are more honest when information is limited

## Common Biases to Avoid

- **Availability bias**: Recent dramatic events feel more likely than they are
- **Confirmation bias**: Seeking evidence that confirms initial hunches
- **Overconfidence**: Giving ranges that are too narrow
- **Anchoring**: Over-weighting the first number you think of
"""

CODER_SYSTEM_PROMPT = """You are an expert software engineer assistant with full tool access. You help with coding tasks, debugging, code review, and software architecture.

## Core Principles

1. **Correctness First**: Working code before clever code.
2. **Simplicity**: Prefer straightforward solutions. Don't over-engineer.
3. **Read Before Write**: Understand existing code before suggesting changes.
4. **Minimal Changes**: Only change what's necessary for the task.
5. **Explain Reasoning**: Explain why, not just what.

## Tool Usage - USE THESE ACTIVELY

You have access to powerful tools. Use them proactively:
- **web_search**: Look up documentation, APIs, libraries, best practices, error messages, Stack Overflow solutions
- **calculate**: Compute complexity (O(n)), memory estimates, performance metrics, array sizes, etc.
- **get_base_rate**: Get historical data on software project success rates, bug frequencies, etc.
- **search_hf_models**: Search HuggingFace for GGUF models (local LLMs)
- **download_hf_model**: Download a GGUF model from HuggingFace to the models folder

IMPORTANT: When asked about current libraries, frameworks, or APIs - ALWAYS use web_search first to get the latest information.

## Response Format

For code questions:
1. If about external APIs/libraries: web_search first
2. Understand the problem/request
3. Provide clear, working code
4. Explain key decisions
5. Note any caveats or edge cases

For debugging:
1. Search for the error message if unfamiliar
2. Identify the likely cause
3. Explain the root cause
4. Provide the fix
5. Explain how to prevent similar issues

## Code Quality Standards

- Clear variable names
- Appropriate comments (explain why, not what)
- Error handling where appropriate
- No unnecessary complexity
- Follow language conventions

## When Unsure

- Use web_search to find answers
- Ask clarifying questions
- State assumptions explicitly
- Provide alternatives with tradeoffs
"""

REASONING_SYSTEM_PROMPT = """You are a highly capable reasoning assistant with access to tools. You excel at complex analysis, multi-step reasoning, research, and problem-solving.

## Core Capabilities

1. **Deep Reasoning**: Break complex problems into steps, consider multiple angles
2. **Research**: Use tools to gather information before answering
3. **Analysis**: Synthesize information from multiple sources
4. **Balanced Perspective**: Present pros/cons, tradeoffs, uncertainties

## Tool Usage - USE THESE ACTIVELY

You have powerful tools. Use them liberally:
- **web_search**: Research topics, find current information, verify facts, explore multiple sources
- **calculate**: Perform calculations, statistics, unit conversions, compound probabilities
- **get_base_rate**: Get historical frequencies for events (useful for predictions, risk assessment)
- **search_hf_models**: Search HuggingFace for GGUF models to download
- **download_hf_model**: Download a GGUF model from HuggingFace

IMPORTANT: For ANY question about current events, recent developments, or factual claims - use web_search FIRST before answering.

## Response Approach

1. **Gather Information**: Use tools to research before forming conclusions
2. **Think Step by Step**: Break down complex questions
3. **Show Your Work**: Explain reasoning transparently
4. **Acknowledge Uncertainty**: Be clear about confidence levels
5. **Multiple Perspectives**: Consider different viewpoints

## Thinking Mode

For complex problems, you may use <think>...</think> tags to reason through the problem before giving your final answer. This helps with:
- Multi-step math problems
- Logic puzzles
- Complex code analysis
- Research synthesis

## When to Use Each Tool

- **web_search**: Current events, technical docs, fact-checking, "what is X", "how does Y work"
- **calculate**: Any math, conversions, statistics, "what is X * Y", percentages
- **get_base_rate**: Predictions, risk assessment, "how often does X happen"
"""


def get_system_prompt(model_name: str) -> str:
    """Get the system prompt for a model mode."""
    prompts = {
        "forecaster": FORECASTER_SYSTEM_PROMPT,
        "coder": CODER_SYSTEM_PROMPT,
        "reasoning": REASONING_SYSTEM_PROMPT,
    }
    return prompts.get(model_name, REASONING_SYSTEM_PROMPT)


def get_tool_definitions(model_name: str, include_external: bool = True) -> list[dict]:
    """Get tool definitions in OpenAI function calling format.

    Args:
        model_name: The model mode (forecaster, coder, reasoning)
        include_external: Whether to include external AI tools (Claude, OpenAI, Manus)
    """
    # Common tools for all modes
    common_tools = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for current information. Use this to find recent news, data, or facts relevant to the task.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results (default: 5)",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Evaluate a mathematical expression. Supports basic arithmetic, functions like sqrt, log, sin, cos, and constants like pi, e.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "The mathematical expression to evaluate"
                        }
                    },
                    "required": ["expression"]
                }
            }
        }
    ]

    # Forecaster-specific tools
    forecaster_tools = [
        {
            "type": "function",
            "function": {
                "name": "get_base_rate",
                "description": "Get the historical base rate for an event type. Returns frequency data and sources for common prediction market event categories.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "event_type": {
                            "type": "string",
                            "description": "Type of event (e.g., 'incumbent_reelection', 'recession_annual', 'startup_failure')"
                        },
                        "context": {
                            "type": "string",
                            "description": "Optional additional context for the lookup"
                        }
                    },
                    "required": ["event_type"]
                }
            }
        }
    ]

    # HuggingFace model tools
    huggingface_tools = [
        {
            "type": "function",
            "function": {
                "name": "search_hf_models",
                "description": "Search HuggingFace for GGUF models. Use this to find downloadable local LLM models by name, size, or purpose.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query (e.g., 'Qwen 7B instruct', 'Llama 3 8B', 'coding model GGUF')"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum results to return (default: 5)",
                            "default": 5
                        },
                        "quantization": {
                            "type": "string",
                            "description": "Filter by quantization level (e.g., 'Q4_K_M', 'Q5_K_M', 'Q8_0')"
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "download_hf_model",
                "description": "Download a GGUF model from HuggingFace to the local models directory. Use search_hf_models first to find the repo_id and filename.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "repo_id": {
                            "type": "string",
                            "description": "HuggingFace repository ID (e.g., 'bartowski/Qwen2.5-7B-Instruct-GGUF')"
                        },
                        "filename": {
                            "type": "string",
                            "description": "GGUF filename to download (e.g., 'Qwen2.5-7B-Instruct-Q5_K_M.gguf')"
                        }
                    },
                    "required": ["repo_id", "filename"]
                }
            }
        }
    ]

    # Polymarket analysis tools
    polymarket_tools = [
        {
            "type": "function",
            "function": {
                "name": "polymarket_search",
                "description": "Search Polymarket prediction markets by keyword. Use to find markets about specific topics.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search term (e.g., 'bitcoin', 'Trump', 'AI', 'election')"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max results (default: 10)",
                            "default": 10
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "polymarket_analyze",
                "description": "Analyze a specific Polymarket market. Get prices, volume, liquidity, and trading signals.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "market": {
                            "type": "string",
                            "description": "Market URL or slug (e.g., 'will-bitcoin-hit-100k-in-2026')"
                        }
                    },
                    "required": ["market"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "polymarket_arbitrage",
                "description": "Scan for arbitrage opportunities where YES + NO tokens cost less than $1 (guaranteed profit).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "min_profit": {
                            "type": "number",
                            "description": "Minimum profit percentage to show (default: 0.5)",
                            "default": 0.5
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "polymarket_positions",
                "description": "Get a user's Polymarket positions and P&L by wallet address.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "wallet_address": {
                            "type": "string",
                            "description": "Ethereum wallet address (0x...)"
                        }
                    },
                    "required": ["wallet_address"]
                }
            }
        }
    ]

    # External AI tools (Claude, OpenAI, Manus)
    external_tools = [
        {
            "type": "function",
            "function": {
                "name": "call_claude",
                "description": "Delegate to Claude for deep reasoning, architecture analysis, or codebase-wide edits. Use when task complexity exceeds local model capabilities, requires multi-file analysis, or needs sophisticated code review.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "Clear description of what Claude should analyze or accomplish"
                        },
                        "context": {
                            "type": "string",
                            "description": "Supporting context, code snippets, or relevant information"
                        },
                        "constraints": {
                            "type": "string",
                            "description": "Any limitations, requirements, or boundaries for the analysis"
                        }
                    },
                    "required": ["task"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "call_openai",
                "description": "Delegate to OpenAI for summarization, user-facing language polish, or content rewriting. Use when producing clean summaries, translating technical content for users, or adjusting tone.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The content to summarize, rewrite, or explain"
                        },
                        "tone": {
                            "type": "string",
                            "enum": ["neutral", "friendly", "professional", "technical", "concise"],
                            "description": "Desired tone for the output (default: neutral)"
                        }
                    },
                    "required": ["content"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "call_manus",
                "description": "REQUIRES USER APPROVAL. Delegate to Manus for controlled computer-use actions (UI automation). Only use when task cannot be done via API or browser automation. All actions are logged and bounded.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "goal": {
                            "type": "string",
                            "description": "Clear description of what should be accomplished"
                        },
                        "allowed_actions": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["click", "type", "scroll", "read", "screenshot", "wait"]
                            },
                            "description": "List of permitted actions (default: read only)"
                        },
                        "constraints": {
                            "type": "string",
                            "description": "Safety constraints and boundaries for the automation"
                        }
                    },
                    "required": ["goal"]
                }
            }
        }
    ]

    # All models get full tool access
    tools = common_tools + forecaster_tools + huggingface_tools + polymarket_tools

    if include_external:
        tools = tools + external_tools

    return tools
