"""
OpenAI API Wrapper
==================
Secure wrapper for OpenAI GPT API.

Use cases: Summarization, user-facing language, rewriting

ISOLATION RULES:
- OpenAI NEVER sees internal system prompts
- OpenAI NEVER sees other API outputs
- OpenAI CANNOT call tools directly
- OpenAI CANNOT persist memory
- All outputs are sanitized before return
"""

import logging
import time
from typing import Optional
from dataclasses import dataclass
from enum import Enum

import openai

from .vault import get_vault, APIProvider

logger = logging.getLogger(__name__)


class OpenAIModel(Enum):
    """Available OpenAI models."""
    GPT4O = "gpt-4o"
    GPT4O_MINI = "gpt-4o-mini"
    GPT4_TURBO = "gpt-4-turbo"


class Tone(Enum):
    """Output tone options."""
    NEUTRAL = "neutral"
    FRIENDLY = "friendly"
    PROFESSIONAL = "professional"
    TECHNICAL = "technical"
    CONCISE = "concise"


@dataclass
class OpenAIRequest:
    """Sanitized request to OpenAI API."""
    content: str
    tone: Tone = Tone.NEUTRAL
    model: OpenAIModel = OpenAIModel.GPT4O_MINI
    max_tokens: int = 2048
    task_type: str = "summarize"  # summarize, rewrite, explain


@dataclass
class OpenAIResponse:
    """Sanitized response from OpenAI API."""
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


class OpenAIAPIWrapper:
    """
    Secure wrapper for OpenAI API calls.

    This wrapper ensures:
    - API key never exposed outside execution
    - Responses sanitized before return
    - No tool/function calling capability
    - No memory persistence
    """

    # Maximum content to process
    MAX_CONTENT_CHARS = 50_000

    # System prompts by task type
    SYSTEM_PROMPTS = {
        "summarize": "You are a summarization assistant. Create clear, concise summaries that preserve key information.",
        "rewrite": "You are a writing assistant. Rewrite content to be clear and well-structured.",
        "explain": "You are an explanation assistant. Make complex content accessible and understandable.",
    }

    # Tone modifiers
    TONE_INSTRUCTIONS = {
        Tone.NEUTRAL: "",
        Tone.FRIENDLY: "Use a warm, approachable tone.",
        Tone.PROFESSIONAL: "Use a formal, professional tone.",
        Tone.TECHNICAL: "Use precise technical language.",
        Tone.CONCISE: "Be extremely brief. Bullet points preferred.",
    }

    def __init__(self):
        """Initialize wrapper."""
        self._client: Optional[openai.OpenAI] = None
        self._call_count = 0
        self._total_tokens_in = 0
        self._total_tokens_out = 0

    def _get_client(self) -> openai.OpenAI:
        """Get or create the API client."""
        if self._client is None:
            vault = get_vault()
            key = vault._get_key(APIProvider.OPENAI)
            if not key:
                raise RuntimeError("OpenAI API key not configured")
            self._client = openai.OpenAI(api_key=key)
        return self._client

    def is_available(self) -> bool:
        """Check if OpenAI API is available."""
        vault = get_vault()
        return vault.is_available(APIProvider.OPENAI)

    def _build_system_prompt(self, request: OpenAIRequest) -> str:
        """Build system prompt based on task and tone."""
        base = self.SYSTEM_PROMPTS.get(request.task_type, self.SYSTEM_PROMPTS["summarize"])
        tone = self.TONE_INSTRUCTIONS.get(request.tone, "")
        if tone:
            return f"{base} {tone}"
        return base

    def _sanitize_input(self, request: OpenAIRequest) -> OpenAIRequest:
        """Sanitize input before sending to API."""
        if len(request.content) > self.MAX_CONTENT_CHARS:
            logger.warning(f"Truncating content from {len(request.content)} to {self.MAX_CONTENT_CHARS}")
            return OpenAIRequest(
                content=request.content[:self.MAX_CONTENT_CHARS] + "\n[TRUNCATED]",
                tone=request.tone,
                model=request.model,
                max_tokens=request.max_tokens,
                task_type=request.task_type
            )
        return request

    def _sanitize_output(self, raw_content: str) -> tuple[str, str]:
        """Sanitize and summarize output."""
        content = raw_content.strip()

        # Create compact summary
        if len(content) <= 300:
            summary = content
        else:
            first_para = content.split('\n\n')[0]
            if len(first_para) <= 300:
                summary = first_para + "..."
            else:
                summary = content[:297] + "..."

        return content, summary

    def call(self, request: OpenAIRequest) -> OpenAIResponse:
        """Make a call to OpenAI API."""
        if not self.is_available():
            return OpenAIResponse(
                success=False,
                content="",
                summary="",
                model_used="",
                error="OpenAI API key not configured"
            )

        request = self._sanitize_input(request)
        start_time = time.time()

        try:
            client = self._get_client()

            # Build user message based on task type
            if request.task_type == "summarize":
                user_msg = f"Summarize the following:\n\n{request.content}"
            elif request.task_type == "rewrite":
                user_msg = f"Rewrite the following to be clearer:\n\n{request.content}"
            elif request.task_type == "explain":
                user_msg = f"Explain the following in simple terms:\n\n{request.content}"
            else:
                user_msg = request.content

            response = client.chat.completions.create(
                model=request.model.value,
                max_tokens=request.max_tokens,
                messages=[
                    {"role": "system", "content": self._build_system_prompt(request)},
                    {"role": "user", "content": user_msg}
                ]
            )

            latency_ms = int((time.time() - start_time) * 1000)

            raw_content = response.choices[0].message.content or ""
            content, summary = self._sanitize_output(raw_content)

            self._call_count += 1
            tokens_in = response.usage.prompt_tokens
            tokens_out = response.usage.completion_tokens
            self._total_tokens_in += tokens_in
            self._total_tokens_out += tokens_out

            logger.info(f"OpenAI call completed: {tokens_in}+{tokens_out} tokens, {latency_ms}ms")

            return OpenAIResponse(
                success=True,
                content=content,
                summary=summary,
                model_used=request.model.value,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                latency_ms=latency_ms
            )

        except openai.AuthenticationError:
            logger.error("OpenAI API authentication failed")
            return OpenAIResponse(
                success=False,
                content="",
                summary="",
                model_used=request.model.value,
                error="Authentication failed - check API key"
            )

        except openai.RateLimitError:
            logger.warning("OpenAI API rate limited")
            return OpenAIResponse(
                success=False,
                content="",
                summary="",
                model_used=request.model.value,
                error="Rate limited - try again later"
            )

        except openai.APIError as e:
            logger.error(f"OpenAI API error: {type(e).__name__}")
            return OpenAIResponse(
                success=False,
                content="",
                summary="",
                model_used=request.model.value,
                error=f"API error: {type(e).__name__}"
            )

        except Exception as e:
            logger.error(f"Unexpected error calling OpenAI: {type(e).__name__}")
            return OpenAIResponse(
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
_wrapper: Optional[OpenAIAPIWrapper] = None


def get_openai_api() -> OpenAIAPIWrapper:
    """Get the OpenAI API wrapper instance."""
    global _wrapper
    if _wrapper is None:
        _wrapper = OpenAIAPIWrapper()
    return _wrapper
