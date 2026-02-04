"""
Moltbot Inference Engine
========================
Handles communication with llama.cpp server, tool calling loop, response parsing.
"""

import asyncio
import json
import logging
import re
from typing import Optional, Callable, Any
from dataclasses import dataclass, field

import aiohttp

from .server import LlamaServer
from .tools import ToolRegistry, ToolResult
from .prompts import get_system_prompt, get_tool_definitions
from .state import ConversationState

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A single message in the conversation."""
    role: str  # 'system', 'user', 'assistant', 'tool'
    content: str
    tool_calls: Optional[list[dict]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None  # Tool name for tool messages


@dataclass
class InferenceResult:
    """Result of an inference call."""
    content: str
    tool_calls: list[dict] = field(default_factory=list)
    finish_reason: str = "stop"
    tokens_used: int = 0


class InferenceEngine:
    """Manages inference with tool calling loop."""

    def __init__(self, server: LlamaServer, tool_registry: ToolRegistry):
        self.server = server
        self.tools = tool_registry
        self.max_tool_rounds = 5  # Prevent infinite loops

    async def chat(
        self,
        user_message: str,
        conversation: ConversationState,
        model_name: Optional[str] = None
    ) -> str:
        """
        Process a user message and return the assistant's response.

        Args:
            user_message: The user's input.
            conversation: Conversation state for context.
            model_name: Override the current model.

        Returns:
            The assistant's final response text.
        """
        # Ensure server is running with correct model
        current_model = model_name or self.server.state.model_loaded
        if not self.server.state.running:
            if not await self.server.start(current_model):
                return "Error: Failed to start inference server."

        # Add user message to conversation
        conversation.add_message(Message(role="user", content=user_message))

        # Build messages for API
        messages = self._build_messages(conversation, current_model)

        # Tool calling loop
        for round_num in range(self.max_tool_rounds):
            result = await self._call_api(messages, current_model)

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
                "content": result.content or "",
                "tool_calls": result.tool_calls
            })

            # Execute tools and add results
            for tool_call in result.tool_calls:
                tool_result = await self._execute_tool(tool_call)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.get("id", ""),
                    "name": tool_call["function"]["name"],
                    "content": tool_result.content
                })

        # Max rounds exceeded
        final_content = result.content or "I've reached the maximum number of tool calls. Here's what I found so far."
        conversation.add_message(Message(role="assistant", content=final_content))
        return final_content

    def _build_messages(self, conversation: ConversationState, model_name: str) -> list[dict]:
        """Build the messages list for the API call."""
        messages = []

        # System prompt
        system_prompt = get_system_prompt(model_name)
        messages.append({"role": "system", "content": system_prompt})

        # Conversation history (already pruned by ConversationState)
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

    async def _call_api(self, messages: list[dict], model_name: str) -> InferenceResult:
        """Make an API call to the llama.cpp server."""
        model_cfg = self.server.config['models'].get(model_name, {})

        # Get tool definitions for this model
        tools = get_tool_definitions(model_name)

        payload = {
            "messages": messages,
            "temperature": model_cfg.get('temperature', 0.3),
            "top_p": model_cfg.get('top_p', 0.9),
            "max_tokens": 2048,
            "stream": False
        }

        # Add tools if available
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.server.api_base}/v1/chat/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.error(f"API error: {resp.status} - {error_text}")
                        return InferenceResult(
                            content=f"Error: Server returned status {resp.status}",
                            finish_reason="error"
                        )

                    data = await resp.json()
                    return self._parse_response(data)

        except asyncio.TimeoutError:
            logger.error("API request timed out")
            return InferenceResult(content="Error: Request timed out", finish_reason="error")
        except Exception as e:
            logger.error(f"API request failed: {e}")
            return InferenceResult(content=f"Error: {str(e)}", finish_reason="error")

    def _parse_response(self, data: dict) -> InferenceResult:
        """Parse the API response."""
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})

        content = message.get("content", "")
        tool_calls = message.get("tool_calls", [])
        finish_reason = choice.get("finish_reason", "stop")

        # Also check for tool calls in content (some models embed them)
        if not tool_calls and content:
            tool_calls = self._extract_tool_calls_from_content(content)

        tokens_used = data.get("usage", {}).get("total_tokens", 0)

        return InferenceResult(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            tokens_used=tokens_used
        )

    def _extract_tool_calls_from_content(self, content: str) -> list[dict]:
        """
        Extract tool calls from content if model embeds them as JSON.
        Handles various formats models might use.
        """
        tool_calls = []

        # Pattern 1: <tool_call>{"name": ..., "arguments": ...}</tool_call>
        pattern1 = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        matches = re.findall(pattern1, content, re.DOTALL)
        for match in matches:
            try:
                call = json.loads(match)
                tool_calls.append({
                    "id": f"call_{len(tool_calls)}",
                    "type": "function",
                    "function": {
                        "name": call.get("name", call.get("function", "")),
                        "arguments": json.dumps(call.get("arguments", call.get("parameters", {})))
                    }
                })
            except json.JSONDecodeError:
                pass

        # Pattern 2: ```json {"function": "...", "arguments": {...}} ```
        pattern2 = r'```json\s*(\{.*?\})\s*```'
        if not tool_calls:
            matches = re.findall(pattern2, content, re.DOTALL)
            for match in matches:
                try:
                    call = json.loads(match)
                    if "function" in call or "name" in call:
                        tool_calls.append({
                            "id": f"call_{len(tool_calls)}",
                            "type": "function",
                            "function": {
                                "name": call.get("function", call.get("name", "")),
                                "arguments": json.dumps(call.get("arguments", call.get("parameters", {})))
                            }
                        })
                except json.JSONDecodeError:
                    pass

        return tool_calls

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
        logger.info(f"Tool result: {result.content[:100]}...")

        return result
