"""
External Intelligence Layer
===========================
Orchestrates external AI APIs with strict security and isolation.

ARCHITECTURE:
- Local model is the SOLE decision-maker
- External APIs are tools, not controllers
- All outputs are sanitized before entering context
- No raw API responses stored long-term
- Vector memory stores only derived knowledge, never secrets

COMPONENTS:
- SecureKeyVault: Manages API keys securely
- ClaudeAPIWrapper: Deep reasoning capabilities
- OpenAIAPIWrapper: Summarization and language
- ManusAPIWrapper: Computer use with safety
- DelegationPolicy: Explicit decision rules
"""

import logging
from typing import Optional, Any
from dataclasses import dataclass
from pathlib import Path

from .vault import (
    SecureKeyVault,
    get_vault,
    require_keys,
    APIProvider,
    KeyStatus
)
from .claude_api import (
    ClaudeAPIWrapper,
    ClaudeRequest,
    ClaudeResponse,
    ClaudeModel,
    get_claude_api
)
from .openai_api import (
    OpenAIAPIWrapper,
    OpenAIRequest,
    OpenAIResponse,
    OpenAIModel,
    Tone,
    get_openai_api
)
from .manus_api import (
    ManusAPIWrapper,
    ManusRequest,
    ManusResponse,
    ManusAction,
    ManusActionLog,
    get_manus_api
)
from .openclaw_api import (
    OpenClawAPIWrapper,
    OpenClawRequest,
    OpenClawResponse,
    get_openclaw_api
)
from .delegation import (
    DelegationPolicy,
    DelegationDecision,
    DelegationTarget,
    TaskCategory,
    TaskAnalysis,
    get_delegation_policy
)

logger = logging.getLogger(__name__)

# Re-export key classes
__all__ = [
    # Vault
    "SecureKeyVault",
    "get_vault",
    "require_keys",
    "APIProvider",
    "KeyStatus",

    # Claude
    "ClaudeAPIWrapper",
    "ClaudeRequest",
    "ClaudeResponse",
    "ClaudeModel",
    "get_claude_api",

    # OpenAI
    "OpenAIAPIWrapper",
    "OpenAIRequest",
    "OpenAIResponse",
    "OpenAIModel",
    "Tone",
    "get_openai_api",

    # Manus
    "ManusAPIWrapper",
    "ManusRequest",
    "ManusResponse",
    "ManusAction",
    "ManusActionLog",
    "get_manus_api",

    # OpenClaw
    "OpenClawAPIWrapper",
    "OpenClawRequest",
    "OpenClawResponse",
    "get_openclaw_api",

    # Delegation
    "DelegationPolicy",
    "DelegationDecision",
    "DelegationTarget",
    "TaskCategory",
    "TaskAnalysis",
    "get_delegation_policy",

    # Orchestrator
    "ExternalIntelligenceLayer",
    "get_external_layer"
]


@dataclass
class ExecutionResult:
    """Result of executing a delegated task."""
    success: bool
    target: DelegationTarget
    content: str
    summary: str  # Compact summary for context
    tokens_used: int
    latency_ms: int
    error: Optional[str] = None


class ExternalIntelligenceLayer:
    """
    Main orchestrator for external AI capabilities.

    The local model calls this layer to:
    1. Decide where to delegate a task
    2. Execute the delegation safely
    3. Get sanitized results back

    INVARIANT: Local model remains in control at all times.
    """

    def __init__(self, log_dir: Optional[Path] = None):
        """Initialize the external intelligence layer."""
        self._vault = get_vault()
        self._claude = get_claude_api()
        self._openai = get_openai_api()
        self._manus = get_manus_api(log_dir)
        self._openclaw = get_openclaw_api()
        self._policy = get_delegation_policy()

        self._execution_count = 0
        self._manus_user_approved = False

        logger.info("External Intelligence Layer initialized")
        logger.info(f"  Claude: {'available' if self._claude.is_available() else 'not configured'}")
        logger.info(f"  OpenAI: {'available' if self._openai.is_available() else 'not configured'}")
        logger.info(f"  Manus: {'available' if self._manus.is_available() else 'not configured'}")
        logger.info(f"  OpenClaw: {'available' if self._openclaw.is_available() else 'not configured'}")

    def get_availability(self) -> dict[str, bool]:
        """Get availability status of all external APIs."""
        return {
            "claude": self._claude.is_available(),
            "openai": self._openai.is_available(),
            "manus": self._manus.is_available(),
            "openclaw": self._openclaw.is_available()
        }

    def validate_startup(self, required: list[APIProvider] = None) -> tuple[bool, list[str]]:
        """Validate required keys are present."""
        return self._vault.validate_startup(required or [])

    def set_manus_approval(self, approved: bool) -> None:
        """Set user approval for Manus computer use."""
        self._manus_user_approved = approved
        self._manus.set_user_approved(approved)

    def trigger_manus_kill_switch(self, reason: str = "Manual trigger") -> None:
        """Emergency stop for Manus."""
        self._manus.trigger_kill_switch(reason)
        self._manus_user_approved = False

    def analyze_task(self, task: str, context: str = "") -> TaskAnalysis:
        """Analyze a task without executing."""
        return self._policy.analyze_task(task, context)

    def decide_delegation(self, task: str, context: str = "") -> DelegationDecision:
        """Get delegation decision without executing."""
        return self._policy.decide(task, context, self._manus_user_approved)

    async def call_claude(
        self,
        task: str,
        context: str = "",
        constraints: str = "",
        model: ClaudeModel = ClaudeModel.SONNET
    ) -> ExecutionResult:
        """
        Call Claude for deep reasoning.

        Args:
            task: What Claude should do
            context: Supporting information
            constraints: Limitations or requirements
            model: Which Claude model to use

        Returns:
            Sanitized ExecutionResult
        """
        request = ClaudeRequest(
            task=task,
            context=context,
            constraints=constraints,
            model=model
        )

        response = self._claude.call(request)
        self._execution_count += 1

        return ExecutionResult(
            success=response.success,
            target=DelegationTarget.CLAUDE,
            content=response.content,
            summary=response.summary,
            tokens_used=response.tokens_in + response.tokens_out,
            latency_ms=response.latency_ms,
            error=response.error
        )

    async def call_openai(
        self,
        content: str,
        tone: Tone = Tone.NEUTRAL,
        task_type: str = "summarize",
        model: OpenAIModel = OpenAIModel.GPT4O_MINI
    ) -> ExecutionResult:
        """
        Call OpenAI for summarization/language tasks.

        Args:
            content: Content to process
            tone: Output tone
            task_type: "summarize", "rewrite", or "explain"
            model: Which OpenAI model to use

        Returns:
            Sanitized ExecutionResult
        """
        request = OpenAIRequest(
            content=content,
            tone=tone,
            task_type=task_type,
            model=model
        )

        response = self._openai.call(request)
        self._execution_count += 1

        return ExecutionResult(
            success=response.success,
            target=DelegationTarget.OPENAI,
            content=response.content,
            summary=response.summary,
            tokens_used=response.tokens_in + response.tokens_out,
            latency_ms=response.latency_ms,
            error=response.error
        )

    async def call_manus(
        self,
        goal: str,
        allowed_actions: list[ManusAction] = None,
        constraints: str = "",
        dry_run: bool = True
    ) -> ExecutionResult:
        """
        Call Manus for computer use (with safety controls).

        Args:
            goal: What Manus should accomplish
            allowed_actions: List of permitted actions
            constraints: Safety constraints
            dry_run: If True, only preview (default True for safety)

        Returns:
            Sanitized ExecutionResult
        """
        if not self._manus_user_approved and not dry_run:
            return ExecutionResult(
                success=False,
                target=DelegationTarget.MANUS,
                content="",
                summary="",
                tokens_used=0,
                latency_ms=0,
                error="User approval required for Manus execution"
            )

        request = ManusRequest(
            goal=goal,
            allowed_actions=allowed_actions or [ManusAction.READ],
            constraints=constraints,
            dry_run=dry_run
        )

        response = await self._manus.call(request)
        self._execution_count += 1

        return ExecutionResult(
            success=response.success,
            target=DelegationTarget.MANUS,
            content=response.result,
            summary=response.summary,
            tokens_used=0,  # Manus doesn't use tokens
            latency_ms=response.latency_ms,
            error=response.error
        )

    async def send_openclaw_message(
        self,
        channel: str,
        target: str,
        message: str
    ) -> ExecutionResult:
        """
        Send a message through OpenClaw.

        Args:
            channel: Channel name (e.g., 'telegram', 'whatsapp', 'discord')
            target: Recipient identifier
            message: Message text to send

        Returns:
            Sanitized ExecutionResult
        """
        response = await self._openclaw.send_message(channel, target, message)
        self._execution_count += 1

        return ExecutionResult(
            success=response.success,
            target=DelegationTarget.LOCAL,  # OpenClaw is a local tool
            content=response.content,
            summary=response.summary,
            tokens_used=0,
            latency_ms=response.latency_ms,
            error=response.error
        )

    async def call_openclaw_agent(
        self,
        message: str,
        thinking: str = "medium"
    ) -> ExecutionResult:
        """
        Delegate a task to the OpenClaw agent.

        Args:
            message: Task or question for the agent
            thinking: Reasoning depth (off, minimal, low, medium, high)

        Returns:
            Sanitized ExecutionResult
        """
        response = await self._openclaw.run_agent(message, thinking)
        self._execution_count += 1

        return ExecutionResult(
            success=response.success,
            target=DelegationTarget.LOCAL,  # OpenClaw is a local tool
            content=response.content,
            summary=response.summary,
            tokens_used=0,
            latency_ms=response.latency_ms,
            error=response.error
        )

    async def execute_delegated(
        self,
        task: str,
        context: str = "",
        constraints: str = "",
        tone: Tone = Tone.NEUTRAL,
        force_target: Optional[DelegationTarget] = None
    ) -> ExecutionResult:
        """
        Execute a task with automatic delegation.

        The policy engine decides where to route the task,
        unless force_target is specified.

        Args:
            task: Task description
            context: Supporting context
            constraints: Task constraints
            tone: Output tone (for OpenAI)
            force_target: Override delegation decision

        Returns:
            Sanitized ExecutionResult
        """
        # Get delegation decision
        if force_target:
            decision = DelegationDecision(
                target=force_target,
                reason="Forced by caller",
                confidence=1.0
            )
        else:
            decision = self._policy.decide(task, context, self._manus_user_approved)

        # Execute based on target
        if decision.target == DelegationTarget.LOCAL:
            # Return indicator that local should handle
            return ExecutionResult(
                success=True,
                target=DelegationTarget.LOCAL,
                content="",
                summary="Task should be handled locally",
                tokens_used=0,
                latency_ms=0
            )

        elif decision.target == DelegationTarget.CLAUDE:
            return await self.call_claude(task, context, constraints)

        elif decision.target == DelegationTarget.OPENAI:
            return await self.call_openai(context or task, tone)

        elif decision.target == DelegationTarget.MANUS:
            if decision.requires_approval and not self._manus_user_approved:
                return ExecutionResult(
                    success=False,
                    target=DelegationTarget.MANUS,
                    content="",
                    summary="Requires user approval",
                    tokens_used=0,
                    latency_ms=0,
                    error="User approval required for computer use"
                )
            return await self.call_manus(task, constraints=constraints)

        else:
            return ExecutionResult(
                success=False,
                target=DelegationTarget.LOCAL,
                content="",
                summary="",
                tokens_used=0,
                latency_ms=0,
                error=f"Unknown delegation target: {decision.target}"
            )

    def get_stats(self) -> dict:
        """Get usage statistics for all APIs."""
        return {
            "total_executions": self._execution_count,
            "claude": self._claude.get_stats(),
            "openai": self._openai.get_stats(),
            "manus": self._manus.get_session_stats(),
            "openclaw": self._openclaw.get_stats(),
            "delegation_log_size": len(self._policy.get_decision_log())
        }


# Module-level instance
_layer: Optional[ExternalIntelligenceLayer] = None


def get_external_layer(log_dir: Optional[Path] = None) -> ExternalIntelligenceLayer:
    """Get the external intelligence layer instance."""
    global _layer
    if _layer is None:
        _layer = ExternalIntelligenceLayer(log_dir)
    return _layer
