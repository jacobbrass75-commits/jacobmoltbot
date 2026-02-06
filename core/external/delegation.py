"""
Delegation Policy Engine
========================
Explicit, codified rules for when to delegate to external APIs.

The LOCAL MODEL is ALWAYS the decision-maker. External APIs are tools,
not controllers. This policy enforces that invariant.

DELEGATION RULES (EXPLICIT, NOT IMPLICIT):

CLAUDE when:
- Task exceeds local reasoning depth
- Multi-file or architectural edits needed
- System behavior must be analyzed
- Complex code review/refactoring

OPENAI when:
- Summarizing large outputs
- Producing clean human-facing responses
- Rewriting or explaining results
- Translation or tone adjustment

MANUS only when:
- Explicit user approval exists
- Task cannot be done via API or browser automation
- Actions are bounded and logged
- No alternative approach available
"""

import logging
from typing import Optional
from dataclasses import dataclass
from enum import Enum, auto

from .vault import get_vault, APIProvider

logger = logging.getLogger(__name__)


class DelegationTarget(Enum):
    """Where to delegate a task."""
    LOCAL = auto()      # Keep it local
    CLAUDE = auto()     # Deep reasoning
    OPENAI = auto()     # Summarization/language
    MANUS = auto()      # Computer use


class TaskCategory(Enum):
    """Categories of tasks for delegation decisions."""
    REASONING = "reasoning"
    CODE_ANALYSIS = "code_analysis"
    CODE_GENERATION = "code_generation"
    SUMMARIZATION = "summarization"
    USER_RESPONSE = "user_response"
    COMPUTER_USE = "computer_use"
    SIMPLE_QUERY = "simple_query"
    CALCULATION = "calculation"
    WEB_SEARCH = "web_search"


@dataclass
class DelegationDecision:
    """Result of delegation policy evaluation."""
    target: DelegationTarget
    reason: str
    confidence: float  # 0.0 to 1.0
    fallback: Optional[DelegationTarget] = None
    requires_approval: bool = False


@dataclass
class TaskAnalysis:
    """Analysis of a task for delegation purposes."""
    category: TaskCategory
    complexity_score: float  # 0.0 to 1.0
    requires_multi_file: bool
    requires_architecture_knowledge: bool
    is_user_facing: bool
    requires_computer_use: bool
    estimated_tokens: int
    keywords: list[str]


class DelegationPolicy:
    """
    Codified delegation policy engine.

    All delegation decisions go through this policy.
    The local model calls this to decide where to send tasks.
    """

    # Complexity thresholds
    LOCAL_COMPLEXITY_THRESHOLD = 0.4  # Below this, keep local
    CLAUDE_COMPLEXITY_THRESHOLD = 0.7  # Above this, use Claude

    # Token thresholds
    LOCAL_TOKEN_LIMIT = 2000  # Simple tasks
    SUMMARIZATION_THRESHOLD = 5000  # Large content to summarize

    # Keywords that suggest specific delegation
    CLAUDE_KEYWORDS = {
        "architecture", "refactor", "design", "analyze", "review",
        "complex", "multi-file", "system", "codebase", "deep"
    }

    OPENAI_KEYWORDS = {
        "summarize", "summary", "explain", "rewrite", "simplify",
        "user-friendly", "translate", "tone", "readable"
    }

    MANUS_KEYWORDS = {
        "click", "browser", "automate", "ui", "gui",
        "desktop", "window", "mouse", "keyboard"
    }

    def __init__(self):
        """Initialize policy engine."""
        self._decision_log: list[dict] = []

    def analyze_task(self, task_description: str, context: str = "") -> TaskAnalysis:
        """
        Analyze a task to determine its characteristics.

        Args:
            task_description: What needs to be done
            context: Additional context (code, previous output, etc.)

        Returns:
            TaskAnalysis with task characteristics
        """
        text = (task_description + " " + context).lower()
        words = set(text.split())

        # Detect category
        category = TaskCategory.SIMPLE_QUERY

        if any(kw in text for kw in ["code", "function", "class", "implement"]):
            if any(kw in text for kw in ["review", "analyze", "refactor"]):
                category = TaskCategory.CODE_ANALYSIS
            else:
                category = TaskCategory.CODE_GENERATION
        elif any(kw in text for kw in self.OPENAI_KEYWORDS):
            if "summarize" in text or "summary" in text:
                category = TaskCategory.SUMMARIZATION
            else:
                category = TaskCategory.USER_RESPONSE
        elif any(kw in text for kw in self.MANUS_KEYWORDS):
            category = TaskCategory.COMPUTER_USE
        elif any(kw in text for kw in ["think", "reason", "complex", "analyze"]):
            category = TaskCategory.REASONING
        elif any(kw in text for kw in ["calculate", "compute", "math"]):
            category = TaskCategory.CALCULATION
        elif any(kw in text for kw in ["search", "find", "look up"]):
            category = TaskCategory.WEB_SEARCH

        # Estimate complexity
        complexity = 0.3  # Base complexity

        # Increase for certain indicators
        if len(context) > 5000:
            complexity += 0.2
        if any(kw in text for kw in self.CLAUDE_KEYWORDS):
            complexity += 0.3
        if "multiple" in text or "several" in text:
            complexity += 0.1

        complexity = min(1.0, complexity)

        # Check for multi-file indicators
        requires_multi_file = any(kw in text for kw in [
            "multiple files", "across files", "codebase", "refactor",
            "all files", "project-wide", "system-wide"
        ])

        # Check for architecture indicators
        requires_architecture = any(kw in text for kw in [
            "architecture", "design", "structure", "pattern",
            "system design", "data flow", "component"
        ])

        # Check if user-facing
        is_user_facing = any(kw in text for kw in [
            "user", "customer", "client", "readable",
            "friendly", "explain to", "present"
        ])

        # Check for computer use
        requires_computer_use = any(kw in text for kw in self.MANUS_KEYWORDS)

        # Estimate tokens
        estimated_tokens = len(text.split()) * 2 + len(context.split())

        # Extract matched keywords
        keywords = list(words & (self.CLAUDE_KEYWORDS | self.OPENAI_KEYWORDS | self.MANUS_KEYWORDS))

        return TaskAnalysis(
            category=category,
            complexity_score=complexity,
            requires_multi_file=requires_multi_file,
            requires_architecture_knowledge=requires_architecture,
            is_user_facing=is_user_facing,
            requires_computer_use=requires_computer_use,
            estimated_tokens=estimated_tokens,
            keywords=keywords
        )

    def decide(self, task: str, context: str = "", user_approved_manus: bool = False) -> DelegationDecision:
        """
        Make a delegation decision for a task.

        Args:
            task: Task description
            context: Additional context
            user_approved_manus: Whether user has approved Manus use

        Returns:
            DelegationDecision with target and reasoning
        """
        analysis = self.analyze_task(task, context)
        vault = get_vault()

        # Rule 1: Computer use ONLY with explicit approval
        if analysis.requires_computer_use:
            if not user_approved_manus:
                decision = DelegationDecision(
                    target=DelegationTarget.LOCAL,
                    reason="Computer use requires explicit user approval",
                    confidence=1.0,
                    requires_approval=True
                )
            elif not vault.is_available(APIProvider.MANUS):
                decision = DelegationDecision(
                    target=DelegationTarget.LOCAL,
                    reason="Manus API not available",
                    confidence=1.0
                )
            else:
                decision = DelegationDecision(
                    target=DelegationTarget.MANUS,
                    reason="Computer use task with user approval",
                    confidence=0.9,
                    requires_approval=True
                )

        # Rule 2: Summarization goes to OpenAI
        elif analysis.category == TaskCategory.SUMMARIZATION:
            if vault.is_available(APIProvider.OPENAI):
                decision = DelegationDecision(
                    target=DelegationTarget.OPENAI,
                    reason="Summarization task - OpenAI specializes in this",
                    confidence=0.85,
                    fallback=DelegationTarget.LOCAL
                )
            else:
                decision = DelegationDecision(
                    target=DelegationTarget.LOCAL,
                    reason="OpenAI not available for summarization",
                    confidence=0.7
                )

        # Rule 3: User-facing response formatting goes to OpenAI
        elif analysis.category == TaskCategory.USER_RESPONSE and analysis.is_user_facing:
            if vault.is_available(APIProvider.OPENAI):
                decision = DelegationDecision(
                    target=DelegationTarget.OPENAI,
                    reason="User-facing response - OpenAI for polish",
                    confidence=0.8,
                    fallback=DelegationTarget.LOCAL
                )
            else:
                decision = DelegationDecision(
                    target=DelegationTarget.LOCAL,
                    reason="OpenAI not available for response formatting",
                    confidence=0.7
                )

        # Rule 4: Complex reasoning/analysis goes to Claude
        elif analysis.complexity_score >= self.CLAUDE_COMPLEXITY_THRESHOLD:
            if vault.is_available(APIProvider.CLAUDE):
                decision = DelegationDecision(
                    target=DelegationTarget.CLAUDE,
                    reason=f"High complexity ({analysis.complexity_score:.2f}) - Claude for deep reasoning",
                    confidence=0.9,
                    fallback=DelegationTarget.LOCAL
                )
            else:
                decision = DelegationDecision(
                    target=DelegationTarget.LOCAL,
                    reason="Claude not available for complex task",
                    confidence=0.5
                )

        # Rule 5: Multi-file or architecture tasks go to Claude
        elif analysis.requires_multi_file or analysis.requires_architecture_knowledge:
            if vault.is_available(APIProvider.CLAUDE):
                decision = DelegationDecision(
                    target=DelegationTarget.CLAUDE,
                    reason="Multi-file/architecture task - Claude for analysis",
                    confidence=0.85,
                    fallback=DelegationTarget.LOCAL
                )
            else:
                decision = DelegationDecision(
                    target=DelegationTarget.LOCAL,
                    reason="Claude not available for architecture task",
                    confidence=0.5
                )

        # Rule 6: Code analysis goes to Claude
        elif analysis.category == TaskCategory.CODE_ANALYSIS:
            if vault.is_available(APIProvider.CLAUDE):
                decision = DelegationDecision(
                    target=DelegationTarget.CLAUDE,
                    reason="Code analysis task - Claude for review",
                    confidence=0.8,
                    fallback=DelegationTarget.LOCAL
                )
            else:
                decision = DelegationDecision(
                    target=DelegationTarget.LOCAL,
                    reason="Claude not available for code analysis",
                    confidence=0.6
                )

        # Rule 7: Simple tasks stay local
        elif analysis.complexity_score < self.LOCAL_COMPLEXITY_THRESHOLD:
            decision = DelegationDecision(
                target=DelegationTarget.LOCAL,
                reason=f"Low complexity ({analysis.complexity_score:.2f}) - local can handle",
                confidence=0.9
            )

        # Rule 8: Default to local with medium confidence
        else:
            decision = DelegationDecision(
                target=DelegationTarget.LOCAL,
                reason="No specific delegation rule matched - keeping local",
                confidence=0.6
            )

        # Log decision
        self._decision_log.append({
            "task": task[:100],
            "analysis": {
                "category": analysis.category.value,
                "complexity": analysis.complexity_score,
                "keywords": analysis.keywords
            },
            "decision": {
                "target": decision.target.name,
                "reason": decision.reason,
                "confidence": decision.confidence
            }
        })

        logger.info(f"Delegation: {decision.target.name} ({decision.confidence:.0%}) - {decision.reason}")

        return decision

    def get_decision_log(self) -> list[dict]:
        """Get the log of all delegation decisions."""
        return self._decision_log.copy()


# Module-level instance
_policy: Optional[DelegationPolicy] = None


def get_delegation_policy() -> DelegationPolicy:
    """Get the delegation policy instance."""
    global _policy
    if _policy is None:
        _policy = DelegationPolicy()
    return _policy
