"""
Moltbot OpenAI Inference Engine
===============================
Uses OpenAI API as the primary inference backend.
"""

import asyncio
import json
import logging
import os
from typing import Optional
from dataclasses import dataclass, field

from openai import AsyncOpenAI

from .tools import ToolRegistry, ToolResult
from .prompts import get_system_prompt, get_tool_definitions
from .state import ConversationState

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A single message in the conversation."""
    role: str
    content: str
    tool_calls: Optional[list[dict]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


@dataclass
class InferenceResult:
    """Result of an inference call."""
    content: str
    tool_calls: list[dict] = field(default_factory=list)
    finish_reason: str = "stop"
    tokens_used: int = 0


class OpenAIInferenceEngine:
    """OpenAI-powered inference with tool calling."""

    # Available models
    MODELS = {
        "gpt-5.2": "gpt-5.2",
        "gpt-5.2-pro": "gpt-5.2-pro",
        "gpt-5": "gpt-5",
        "gpt-4o": "gpt-4o",
        "gpt-4o-mini": "gpt-4o-mini",
        "o1": "o1",
        "o1-mini": "o1-mini",
        "default": "gpt-5.2"
    }

    def __init__(self, tool_registry: ToolRegistry, model: str = "gpt-4o"):
        self.tools = tool_registry
        self.current_model = model
        self.max_tool_rounds = 10
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        logger.info(f"OpenAI inference engine initialized with model: {model}")

    def set_model(self, model: str) -> bool:
        """Switch to a different model."""
        if model in self.MODELS:
            self.current_model = self.MODELS[model]
            logger.info(f"Switched to model: {self.current_model}")
            return True
        elif model.startswith("gpt-") or model.startswith("o1"):
            self.current_model = model
            logger.info(f"Switched to model: {self.current_model}")
            return True
        else:
            logger.warning(f"Unknown model: {model}")
            return False

    async def chat(
        self,
        user_message: str,
        conversation: ConversationState,
        model_name: Optional[str] = None
    ) -> str:
        """
        Process a user message and return the assistant's response.
        """
        # Use specified model or current default
        model = model_name if model_name and model_name.startswith("gpt") else self.current_model

        # Add user message to conversation
        conversation.add_message(Message(role="user", content=user_message))

        # Build messages for API
        messages = self._build_messages(conversation)

        # Get tool definitions
        tools = get_tool_definitions("reasoning")  # Use reasoning tools (most complete)

        # Tool calling loop
        for round_num in range(self.max_tool_rounds):
            try:
                result = await self._call_api(messages, model, tools)
            except Exception as e:
                logger.error(f"API call failed: {e}")
                return f"Error: {str(e)}"

            if not result.tool_calls:
                # No tool calls, we're done
                conversation.add_message(Message(
                    role="assistant",
                    content=result.content
                ))
                return result.content

            # Process tool calls
            logger.info(f"Processing {len(result.tool_calls)} tool call(s) (round {round_num + 1})")

            # Add assistant message with tool calls
            messages.append({
                "role": "assistant",
                "content": result.content or None,
                "tool_calls": result.tool_calls
            })

            # Execute tools and add results
            for tool_call in result.tool_calls:
                tool_result = await self._execute_tool(tool_call)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.get("id", ""),
                    "content": tool_result.content
                })

        # Max rounds exceeded
        final_content = result.content or "I've reached the maximum number of tool calls."
        conversation.add_message(Message(role="assistant", content=final_content))
        return final_content

    def _build_messages(self, conversation: ConversationState) -> list[dict]:
        """Build the messages list for the API call."""
        messages = []

        # System prompt
        system_prompt = get_system_prompt("reasoning")
        messages.append({"role": "system", "content": system_prompt})

        # Conversation history
        for msg in conversation.get_messages():
            msg_dict = {"role": msg.role, "content": msg.content}
            if msg.tool_calls:
                msg_dict["tool_calls"] = msg.tool_calls
            if msg.tool_call_id:
                msg_dict["tool_call_id"] = msg.tool_call_id
            if msg.name:
                msg_dict["name"] = msg.name
            messages.append(msg_dict)

        return messages

    async def _call_api(
        self,
        messages: list[dict],
        model: str,
        tools: list[dict]
    ) -> InferenceResult:
        """Make an API call to OpenAI."""
        try:
            # Filter out None content for assistant messages with tool_calls
            clean_messages = []
            for m in messages:
                if m.get("role") == "assistant" and m.get("tool_calls") and m.get("content") is None:
                    clean_messages.append({
                        "role": "assistant",
                        "content": "",
                        "tool_calls": m["tool_calls"]
                    })
                else:
                    clean_messages.append(m)

            # Build API parameters - newer models use max_completion_tokens
            api_params = {
                "model": model,
                "messages": clean_messages,
                "tools": tools if tools else None,
                "tool_choice": "auto" if tools else None,
            }

            # GPT-5.x and O1 models use max_completion_tokens, others use max_tokens
            if model.startswith("gpt-5") or model.startswith("o1"):
                api_params["max_completion_tokens"] = 4096
            else:
                api_params["temperature"] = 0.7
                api_params["max_tokens"] = 4096

            response = await self.client.chat.completions.create(**api_params)

            choice = response.choices[0]
            message = choice.message

            # Extract tool calls
            tool_calls = []
            if message.tool_calls:
                for tc in message.tool_calls:
                    tool_calls.append({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    })

            return InferenceResult(
                content=message.content or "",
                tool_calls=tool_calls,
                finish_reason=choice.finish_reason or "stop",
                tokens_used=response.usage.total_tokens if response.usage else 0
            )

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    async def _execute_tool(self, tool_call: dict) -> ToolResult:
        """Execute a tool call and return the result."""
        func_info = tool_call.get("function", {})
        tool_name = func_info.get("name", "")
        args_str = func_info.get("arguments", "{}")

        try:
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
        except json.JSONDecodeError:
            return ToolResult(success=False, content=f"Invalid arguments: {args_str}")

        logger.info(f"Executing tool: {tool_name} with args: {args}")

        result = await self.tools.execute(tool_name, args)

        # Truncate very long results
        content = result.content
        if len(content) > 10000:
            content = content[:10000] + "\n... (truncated)"

        logger.info(f"Tool result: {content[:100]}...")

        return ToolResult(success=result.success, content=content, error=result.error)

    def get_status(self) -> dict:
        """Get current engine status."""
        return {
            "running": True,
            "model": self.current_model,
            "backend": "openai",
            "available_models": list(self.MODELS.keys())
        }
