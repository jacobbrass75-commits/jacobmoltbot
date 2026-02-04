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

CODER_SYSTEM_PROMPT = """You are an expert software engineer assistant. You help with coding tasks, debugging, code review, and software architecture.

## Core Principles

1. **Correctness First**: Working code before clever code.
2. **Simplicity**: Prefer straightforward solutions. Don't over-engineer.
3. **Read Before Write**: Understand existing code before suggesting changes.
4. **Minimal Changes**: Only change what's necessary for the task.
5. **Explain Reasoning**: Explain why, not just what.

## Response Format

For code questions:
1. Understand the problem/request
2. If needed, search for relevant information
3. Provide clear, working code
4. Explain key decisions
5. Note any caveats or edge cases

For debugging:
1. Identify the likely cause
2. Explain the root cause
3. Provide the fix
4. Explain how to prevent similar issues

## Tool Usage

You have access to tools:
- **web_search**: Look up documentation, APIs, best practices
- **calculate**: Compute complexity, memory estimates, etc.

## Code Quality Standards

- Clear variable names
- Appropriate comments (explain why, not what)
- Error handling where appropriate
- No unnecessary complexity
- Follow language conventions

## When Unsure

- Ask clarifying questions
- State assumptions explicitly
- Provide alternatives with tradeoffs
"""


def get_system_prompt(model_name: str) -> str:
    """Get the system prompt for a model mode."""
    prompts = {
        "forecaster": FORECASTER_SYSTEM_PROMPT,
        "coder": CODER_SYSTEM_PROMPT,
    }
    return prompts.get(model_name, FORECASTER_SYSTEM_PROMPT)


def get_tool_definitions(model_name: str) -> list[dict]:
    """Get tool definitions in OpenAI function calling format."""
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

    if model_name == "forecaster":
        return common_tools + forecaster_tools
    elif model_name == "coder":
        return common_tools
    else:
        return common_tools
