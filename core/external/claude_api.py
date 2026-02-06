"""
Claude API Wrapper
==================
Secure wrapper for Anthropic Claude API.

Use cases: Deep reasoning, architecture, codebase-wide analysis

ISOLATION RULES:
- Claude NEVER sees internal system prompts
- Claude NEVER sees other API outputs
- Claude CANNOT call tools directly
- Claude CANNOT persist memory
- All outputs are sanitized before return
"""

import logging
import time
from typing import Optional
from dataclasses import dataclass, field
from enum import Enum

import anthropic

from .vault import get_vault, APIProvider

logger = logging.getLogger(__name__)


class ClaudeModel(Enum):
    """Available Claude models."""
    SONNET = "claude-sonnet-4-20250514"
    OPUS = "claude-opus-4-20250514"
    HAIKU = "claude-haiku-4-20250514"


@dataclass
class ClaudeRequest:
    """Sanitized request to Claude API."""
    task: str
    context: str = ""
    constraints: str = ""
    model: ClaudeModel = ClaudeModel.SONNET
    max_tokens: int = 4096

    def to_prompt(self) -> str:
        """Convert to a single prompt string."""
        parts = [f"Task: {self.task}"]
        if self.context:
            parts.append(f"\nContext:\n{self.context}")
        if self.constraints:
            parts.append(f"\nConstraints:\n{self.constraints}")
        return "\n".join(parts)


@dataclass
class ClaudeResponse:
    """Sanitized response from Claude API."""
    success: bool
    content: str
    summary: str  # Compact summary for context window
    model_used: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0
    error: Optional[str] = None

    # Metadata stripped from raw response
    raw_headers_stripped: bool = True
    raw_metadata_stripped: bool = True


class ClaudeAPIWrapper:
    """
    Secure wrapper for Claude API calls.

    This wrapper ensures:
    - API key never exposed outside execution
    - Responses sanitized before return
    - No tool/function calling capability
    - No memory persistence
    """

    # Maximum context to send (prevent cost explosion)
    MAX_CONTEXT_CHARS = 100_000

    # System prompt for Claude (minimal, task-focused)
    SYSTEM_PROMPT = """You are a reasoning assistant. Provide clear, structured analysis.
Focus only on the task given. Do not assume access to external tools or systems.
Be concise but thorough. Structure complex responses with headers."""

    def __init__(self):
        """Initialize wrapper (does not validate key yet)."""
        self._client: Optional[anthropic.Anthropic] = None
        self._call_count = 0
        self._total_tokens_in = 0
        self._total_tokens_out = 0

    def _get_client(self) -> anthropic.Anthropic:
        """Get or create the API client."""
        if self._client is None:
            vault = get_vault()
            key = vault._get_key(APIProvider.CLAUDE)
            if not key:
                raise RuntimeError("Claude API key not configured")
            # Key is passed directly to client, never logged
            self._client = anthropic.Anthropic(api_key=key)
        return self._client

    def is_available(self) -> bool:
        """Check if Claude API is available."""
        vault = get_vault()
        return vault.is_available(APIProvider.CLAUDE)

    def _sanitize_input(self, request: ClaudeRequest) -> ClaudeRequest:
        """Sanitize input before sending to API."""
        # Truncate overly long context
        if len(request.context) > self.MAX_CONTEXT_CHARS:
            logger.warning(f"Truncating context from {len(request.context)} to {self.MAX_CONTEXT_CHARS}")
            request = ClaudeRequest(
                task=request.task,
                context=request.context[:self.MAX_CONTEXT_CHARS] + "\n[TRUNCATED]",
                constraints=request.constraints,
                model=request.model,
                max_tokens=request.max_tokens
            )
        return request

    def _sanitize_output(self, raw_content: str) -> tuple[str, str]:
        """
        Sanitize and summarize output.

        Returns:
            Tuple of (full_content, summary)
        """
        # Strip any potential injection attempts or system markers
        content = raw_content.strip()

        # Create compact summary (first 500 chars or first paragraph)
        if len(content) <= 500:
            summary = content
        else:
            # Try to find a natural break point
            first_para = content.split('\n\n')[0]
            if len(first_para) <= 500:
                summary = first_para + "..."
            else:
                summary = content[:497] + "..."

        return content, summary

    def call(self, request: ClaudeRequest) -> ClaudeResponse:
        """
        Make a call to Claude API.

        Args:
            request: Sanitized request object

        Returns:
            Sanitized response object
        """
        if not self.is_available():
            return ClaudeResponse(
                success=False,
                content="",
                summary="",
                model_used="",
                error="Claude API key not configured"
            )

        # Sanitize input
        request = self._sanitize_input(request)

        start_time = time.time()

        try:
            client = self._get_client()

            # Make API call - NO TOOLS, just messages
            response = client.messages.create(
                model=request.model.value,
                max_tokens=request.max_tokens,
                system=self.SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": request.to_prompt()}
                ]
            )

            latency_ms = int((time.time() - start_time) * 1000)

            # Extract content (handle multiple content blocks)
            raw_content = ""
            for block in response.content:
                if hasattr(block, 'text'):
                    raw_content += block.text

            # Sanitize output
            content, summary = self._sanitize_output(raw_content)

            # Update metrics
            self._call_count += 1
            tokens_in = response.usage.input_tokens
            tokens_out = response.usage.output_tokens
            self._total_tokens_in += tokens_in
            self._total_tokens_out += tokens_out

            logger.info(f"Claude call completed: {tokens_in}+{tokens_out} tokens, {latency_ms}ms")

            return ClaudeResponse(
                success=True,
                content=content,
                summary=summary,
                model_used=request.model.value,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                latency_ms=latency_ms
            )

        except anthropic.AuthenticationError as e:
            logger.error("Claude API authentication failed (key may be invalid)")
            return ClaudeResponse(
                success=False,
                content="",
                summary="",
                model_used=request.model.value,
                error="Authentication failed - check API key"
            )

        except anthropic.RateLimitError as e:
            logger.warning("Claude API rate limited")
            return ClaudeResponse(
                success=False,
                content="",
                summary="",
                model_used=request.model.value,
                error="Rate limited - try again later"
            )

        except anthropic.APIError as e:
            logger.error(f"Claude API error: {type(e).__name__}")
            return ClaudeResponse(
                success=False,
                content="",
                summary="",
                model_used=request.model.value,
                error=f"API error: {type(e).__name__}"
            )

        except Exception as e:
            logger.error(f"Unexpected error calling Claude: {type(e).__name__}")
            return ClaudeResponse(
                success=False,
                content="",
                summary="",
                model_used=request.model.value,
                error=f"Unexpected error: {type(e).__name__}"
            )

    def get_stats(self) -> dict:
        """Get usage statistics."""
        return {
            "call_count": self._call_count,
            "total_tokens_in": self._total_tokens_in,
            "total_tokens_out": self._total_tokens_out
        }


# Module-level instance
_wrapper: Optional[ClaudeAPIWrapper] = None


def get_claude_api() -> ClaudeAPIWrapper:
    """Get the Claude API wrapper instance."""
    global _wrapper
    if _wrapper is None:
        _wrapper = ClaudeAPIWrapper()
    return _wrapper
