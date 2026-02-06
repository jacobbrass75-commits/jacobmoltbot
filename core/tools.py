"""
Moltbot Tool Registry
=====================
Defines and executes tools that the model can call.
"""

import asyncio
import logging
import math
from typing import Callable, Any, Optional
from dataclasses import dataclass
from functools import wraps

from simpleeval import simple_eval, EvalWithCompoundTypes
from duckduckgo_search import DDGS

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result of a tool execution."""
    success: bool
    content: str
    error: Optional[str] = None


class ToolRegistry:
    """Registry of available tools."""

    def __init__(self):
        self._tools: dict[str, Callable] = {}
        self._register_builtin_tools()

    def _register_builtin_tools(self) -> None:
        """Register the built-in tools."""
        self.register("web_search", self._web_search)
        self.register("calculate", self._calculate)
        self.register("get_base_rate", self._get_base_rate)
        self.register("search_hf_models", self._search_hf_models)
        self.register("download_hf_model", self._download_hf_model)
        self.register("polymarket_search", self._polymarket_search)
        self.register("polymarket_analyze", self._polymarket_analyze)
        self.register("polymarket_arbitrage", self._polymarket_arbitrage)
        self.register("polymarket_positions", self._polymarket_positions)
        self.register("openclaw_send_message", self._openclaw_send_message)
        self.register("openclaw_agent", self._openclaw_agent)
        self.register("openclaw_status", self._openclaw_status)
        self.register("openclaw_audit", self._openclaw_audit)

    def register(self, name: str, func: Callable) -> None:
        """Register a tool function."""
        self._tools[name] = func
        logger.debug(f"Registered tool: {name}")

    async def execute(self, name: str, args: dict) -> ToolResult:
        """Execute a tool by name with given arguments."""
        if name not in self._tools:
            return ToolResult(
                success=False,
                content=f"Unknown tool: {name}",
                error="tool_not_found"
            )

        try:
            func = self._tools[name]
            if asyncio.iscoroutinefunction(func):
                result = await func(**args)
            else:
                result = await asyncio.to_thread(func, **args)
            return ToolResult(success=True, content=str(result))
        except Exception as e:
            logger.error(f"Tool execution failed: {name} - {e}")
            return ToolResult(
                success=False,
                content=f"Tool error: {str(e)}",
                error=str(e)
            )

    def _web_search(self, query: str, max_results: int = 5) -> str:
        """
        Search the web using DuckDuckGo.

        Args:
            query: Search query string.
            max_results: Maximum number of results to return.

        Returns:
            Formatted search results.
        """
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))

            if not results:
                return "No search results found."

            formatted = []
            for i, r in enumerate(results, 1):
                formatted.append(
                    f"{i}. **{r.get('title', 'No title')}**\n"
                    f"   {r.get('body', 'No description')}\n"
                    f"   Source: {r.get('href', 'Unknown')}"
                )

            return "\n\n".join(formatted)

        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return f"Search failed: {str(e)}"

    def _calculate(self, expression: str) -> str:
        """
        Safely evaluate a mathematical expression.

        Args:
            expression: Mathematical expression to evaluate.

        Returns:
            The result of the calculation.
        """
        # Safe math functions
        safe_functions = {
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "len": len,
            "pow": pow,
            "sqrt": math.sqrt,
            "log": math.log,
            "log10": math.log10,
            "log2": math.log2,
            "exp": math.exp,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "asin": math.asin,
            "acos": math.acos,
            "atan": math.atan,
            "ceil": math.ceil,
            "floor": math.floor,
        }

        safe_names = {
            "pi": math.pi,
            "e": math.e,
            "inf": float('inf'),
        }

        try:
            evaluator = EvalWithCompoundTypes(
                functions=safe_functions,
                names=safe_names
            )
            result = evaluator.eval(expression)
            return f"{expression} = {result}"
        except Exception as e:
            return f"Calculation error: {str(e)}"

    def _get_base_rate(self, event_type: str, context: Optional[str] = None) -> str:
        """
        Get historical base rate for an event type.

        Args:
            event_type: Type of event to look up.
            context: Optional context for more specific lookup.

        Returns:
            Base rate information and sources.
        """
        # Curated base rates for common prediction market events
        base_rates = {
            # Political
            "incumbent_reelection": {
                "rate": "~70%",
                "source": "Historical US presidential elections since 1900",
                "notes": "Incumbents win ~70% of reelection bids in normal economic conditions"
            },
            "party_retention": {
                "rate": "~50%",
                "source": "US presidential elections post-WW2",
                "notes": "Party retains White House ~50% of the time after 8 years"
            },
            "midterm_loss": {
                "rate": "~90%",
                "source": "US midterm elections",
                "notes": "President's party loses House seats ~90% of midterms"
            },

            # Economic
            "recession_annual": {
                "rate": "~15%",
                "source": "NBER recession data 1945-present",
                "notes": "US enters recession in any given year ~15% of the time"
            },
            "market_crash_10pct": {
                "rate": "~30%",
                "source": "S&P 500 history",
                "notes": "10%+ drawdown occurs in ~30% of calendar years"
            },
            "market_crash_20pct": {
                "rate": "~10%",
                "source": "S&P 500 history",
                "notes": "20%+ drawdown (bear market) occurs in ~10% of years"
            },

            # Technology
            "startup_failure": {
                "rate": "~90%",
                "source": "Startup Genome Report",
                "notes": "~90% of startups fail within 10 years"
            },
            "major_tech_outage": {
                "rate": "~5-10%",
                "source": "Major cloud provider incident reports",
                "notes": "Major outage affecting millions happens 1-2x per year per provider"
            },

            # Geopolitical
            "interstate_war": {
                "rate": "~1-2%",
                "source": "Correlates of War Project",
                "notes": "New interstate war involving major power starts ~1-2% of years"
            },
            "coup_attempt": {
                "rate": "~2%",
                "source": "Powell & Thyne coup dataset",
                "notes": "~2% of country-years see coup attempts globally"
            },

            # Science/Health
            "drug_approval": {
                "rate": "~10%",
                "source": "FDA clinical trial data",
                "notes": "~10% of drugs entering Phase 1 reach FDA approval"
            },
            "pandemic_novel": {
                "rate": "~1%",
                "source": "Historical pandemic frequency",
                "notes": "Novel pandemic pathogen emerges ~1% of years"
            },

            # Default for unknown events
            "unknown": {
                "rate": "Unknown",
                "source": "No curated data available",
                "notes": "Consider searching for historical frequency data"
            }
        }

        # Normalize lookup key
        key = event_type.lower().replace(" ", "_").replace("-", "_")

        # Try exact match
        if key in base_rates:
            data = base_rates[key]
        else:
            # Try partial match
            matched = None
            for k in base_rates:
                if k in key or key in k:
                    matched = base_rates[k]
                    break
            data = matched or base_rates["unknown"]

        result = f"**Base Rate for '{event_type}':**\n"
        result += f"- Rate: {data['rate']}\n"
        result += f"- Source: {data['source']}\n"
        result += f"- Notes: {data['notes']}"

        if context:
            result += f"\n- Context provided: {context}"
            result += "\n\n*Note: Adjust base rate based on specific context factors.*"

        return result

    def _search_hf_models(
        self,
        query: str,
        limit: int = 5,
        quantization: Optional[str] = None
    ) -> str:
        """
        Search HuggingFace for GGUF models.

        Args:
            query: Search query (e.g., "Qwen 7B instruct")
            limit: Maximum results to return
            quantization: Filter by quantization (e.g., "Q4_K_M", "Q5_K_M")

        Returns:
            Formatted search results
        """
        try:
            from .huggingface import HuggingFaceClient

            client = HuggingFaceClient()
            results = client.search_gguf_models(query, limit=limit)

            if not results:
                return "No GGUF models found matching your query."

            formatted = []
            for i, r in enumerate(results, 1):
                files = r.gguf_files
                if quantization:
                    files = [f for f in files if f.quantization and
                            quantization.upper() in f.quantization.upper()]

                if not files:
                    continue

                file_list = ", ".join(f.filename for f in files[:5])
                if len(files) > 5:
                    file_list += f" (+{len(files) - 5} more)"

                formatted.append(
                    f"{i}. **{r.model_name}** by {r.author}\n"
                    f"   Repo: `{r.repo_id}`\n"
                    f"   Downloads: {r.downloads:,} | Likes: {r.likes}\n"
                    f"   GGUF files: {file_list}"
                )

            return "\n\n".join(formatted) if formatted else "No matching GGUF files found."

        except Exception as e:
            logger.error(f"HuggingFace search failed: {e}")
            return f"Search failed: {str(e)}"

    def _download_hf_model(
        self,
        repo_id: str,
        filename: str
    ) -> str:
        """
        Download a GGUF model from HuggingFace.

        Args:
            repo_id: HuggingFace repo (e.g., "bartowski/Qwen2.5-7B-Instruct-GGUF")
            filename: GGUF file to download (e.g., "Qwen2.5-7B-Instruct-Q5_K_M.gguf")

        Returns:
            Status message
        """
        from pathlib import Path

        try:
            from .huggingface import HuggingFaceClient

            models_dir = Path(__file__).parent.parent / "models"
            client = HuggingFaceClient()

            target = models_dir / Path(filename).name
            if target.exists():
                return f"Model already exists: {target}"

            info = client.get_model_info(repo_id, filename)
            if info and info.size_gb:
                size_str = f" ({info.size_gb:.1f} GB)"
            else:
                size_str = ""

            path = client.download_model(repo_id, filename, models_dir)

            return f"Downloaded successfully{size_str}: {path}"

        except Exception as e:
            logger.error(f"Download failed: {e}")
            return f"Download failed: {str(e)}"

    def _polymarket_search(self, query: str, limit: int = 10) -> str:
        """
        Search Polymarket for prediction markets.

        Args:
            query: Search term (e.g., "bitcoin", "election")
            limit: Max results

        Returns:
            Matching markets
        """
        try:
            from .polymarket import search_markets
            return search_markets(query, limit)
        except Exception as e:
            logger.error(f"Polymarket search failed: {e}")
            return f"Search failed: {str(e)}"

    def _polymarket_analyze(self, market: str) -> str:
        """
        Analyze a specific Polymarket market.

        Args:
            market: Market URL or slug

        Returns:
            Market analysis with signals
        """
        try:
            from .polymarket import analyze_market
            return analyze_market(market)
        except Exception as e:
            logger.error(f"Polymarket analysis failed: {e}")
            return f"Analysis failed: {str(e)}"

    def _polymarket_arbitrage(self, min_profit: float = 0.5) -> str:
        """
        Find Polymarket arbitrage opportunities.

        Args:
            min_profit: Minimum profit percentage

        Returns:
            List of arbitrage opportunities
        """
        try:
            from .polymarket import find_arbitrage
            return find_arbitrage(min_profit)
        except Exception as e:
            logger.error(f"Polymarket arbitrage scan failed: {e}")
            return f"Scan failed: {str(e)}"

    def _polymarket_positions(self, wallet_address: str) -> str:
        """
        Get a user's Polymarket positions.

        Args:
            wallet_address: Ethereum wallet (0x...)

        Returns:
            User's positions and P&L
        """
        try:
            from .polymarket import get_user_positions
            return get_user_positions(wallet_address)
        except Exception as e:
            logger.error(f"Polymarket positions failed: {e}")
            return f"Failed: {str(e)}"

    async def _openclaw_send_message(self, channel: str, target: str, message: str) -> str:
        """
        Send a message through OpenClaw to any connected channel.

        Args:
            channel: Channel name (e.g., 'telegram', 'whatsapp', 'discord')
            target: Recipient identifier (phone number, username, channel ID)
            message: Message text to send

        Returns:
            Result of the send operation
        """
        try:
            from .external.openclaw_api import get_openclaw_api

            api = get_openclaw_api()
            if not api.is_available():
                return "OpenClaw is not available. Check that the CLI is installed and gateway is configured."

            response = await api.send_message(channel, target, message)
            if response.success:
                return response.content or "Message sent successfully."
            else:
                return f"Send failed: {response.error}"
        except Exception as e:
            logger.error(f"OpenClaw send_message failed: {e}")
            return f"Failed: {str(e)}"

    async def _openclaw_agent(self, message: str, thinking: str = "medium") -> str:
        """
        Delegate a task to the OpenClaw agent.

        Args:
            message: Task or question for the OpenClaw agent
            thinking: Reasoning depth (off, minimal, low, medium, high)

        Returns:
            Agent result
        """
        try:
            from .external.openclaw_api import get_openclaw_api

            api = get_openclaw_api()
            if not api.is_available():
                return "OpenClaw is not available. Check that the CLI is installed and gateway is configured."

            response = await api.run_agent(message, thinking)
            if response.success:
                return response.content
            else:
                return f"Agent failed: {response.error}"
        except Exception as e:
            logger.error(f"OpenClaw agent failed: {e}")
            return f"Failed: {str(e)}"

    async def _openclaw_status(self) -> str:
        """
        Check OpenClaw gateway and channel health.

        Returns:
            Status information
        """
        try:
            from .external.openclaw_api import get_openclaw_api

            api = get_openclaw_api()
            if not api.is_available():
                return "OpenClaw is not available. Check that the CLI is installed and gateway is configured."

            response = await api.get_status()
            if response.success:
                return response.content
            else:
                return f"Status check failed: {response.error}"
        except Exception as e:
            logger.error(f"OpenClaw status failed: {e}")
            return f"Failed: {str(e)}"

    async def _openclaw_audit(self, auto_fix: bool = False) -> str:
        """
        Run a comprehensive OpenClaw security audit on the local machine.

        Args:
            auto_fix: If True, also apply automatic fixes

        Returns:
            Full audit results
        """
        try:
            from .external.openclaw_api import get_openclaw_api

            api = get_openclaw_api()
            if not api.is_available():
                return "OpenClaw is not available. Check that the CLI is installed and gateway is configured."

            response = await api.run_security_audit(auto_fix=auto_fix)
            if response.success:
                return response.content
            else:
                return f"Audit failed: {response.error}"
        except Exception as e:
            logger.error(f"OpenClaw audit failed: {e}")
            return f"Failed: {str(e)}"

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())
