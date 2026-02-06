"""
Moltbot Polymarket Integration
==============================
Analyze Polymarket prediction markets - arbitrage, whale tracking, user profiles.
"""

import json
import logging
import re
from typing import Optional
from urllib.request import urlopen, Request
from urllib.error import URLError

logger = logging.getLogger(__name__)

GAMMA_API = "https://gamma-api.polymarket.com"
DATA_API = "https://data-api.polymarket.com"


def _fetch(url: str) -> Optional[dict | list]:
    """Fetch JSON from URL."""
    try:
        req = Request(url, headers={"User-Agent": "Moltbot/1.0"})
        with urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except (URLError, json.JSONDecodeError) as e:
        logger.error(f"API error: {e}")
        return None


def search_markets(query: str, limit: int = 10) -> str:
    """
    Search Polymarket for markets matching a query.

    Args:
        query: Search term (e.g., "bitcoin", "election", "AI")
        limit: Max results to return

    Returns:
        Formatted market list
    """
    data = _fetch(f"{GAMMA_API}/markets?active=true&closed=false&limit=100")
    if not data:
        return "Failed to fetch markets."

    query_lower = query.lower()
    matches = []

    for m in data:
        question = m.get("question", "")
        if query_lower in question.lower():
            prices = m.get("outcomePrices", [])
            if isinstance(prices, str):
                prices = json.loads(prices) if prices.startswith("[") else []

            yes_price = float(prices[0]) if len(prices) > 0 else 0
            no_price = float(prices[1]) if len(prices) > 1 else 0
            volume = float(m.get("volume", 0) or 0)

            matches.append({
                "question": question,
                "slug": m.get("slug", ""),
                "yes": yes_price,
                "no": no_price,
                "pair_cost": yes_price + no_price,
                "volume": volume
            })

    matches = sorted(matches, key=lambda x: x["volume"], reverse=True)[:limit]

    if not matches:
        return f"No markets found matching '{query}'."

    lines = [f"**Polymarket Search: '{query}'**\n"]
    for i, m in enumerate(matches, 1):
        arb_flag = " [ARB]" if m["pair_cost"] < 0.98 else ""
        lines.append(
            f"{i}. **{m['question'][:60]}**{arb_flag}\n"
            f"   YES: ${m['yes']:.3f} | NO: ${m['no']:.3f} | Vol: ${m['volume']:,.0f}"
        )

    return "\n".join(lines)


def analyze_market(market_url_or_slug: str) -> str:
    """
    Analyze a specific Polymarket market.

    Args:
        market_url_or_slug: Market URL or slug

    Returns:
        Detailed market analysis
    """
    # Extract slug from URL if needed
    slug = market_url_or_slug
    if "polymarket.com" in market_url_or_slug:
        match = re.search(r'polymarket\.com/event/[^/]+/([^/\?\#]+)', market_url_or_slug)
        if match:
            slug = match.group(1)
        else:
            match = re.search(r'polymarket\.com/event/([^/\?\#]+)', market_url_or_slug)
            if match:
                slug = match.group(1)

    # Try to find market
    data = _fetch(f"{GAMMA_API}/markets?slug={slug}")
    if not data or not isinstance(data, list) or len(data) == 0:
        # Try searching
        data = _fetch(f"{GAMMA_API}/markets?active=true&limit=50")
        if data:
            for m in data:
                if slug.lower() in str(m.get("slug", "")).lower() or slug.lower() in str(m.get("question", "")).lower():
                    data = [m]
                    break
            else:
                return f"Market not found: {slug}"
        else:
            return "Failed to fetch market data."

    market = data[0] if isinstance(data, list) else data

    # Parse data
    question = market.get("question", "Unknown")
    prices = market.get("outcomePrices", [])
    if isinstance(prices, str):
        prices = json.loads(prices) if prices.startswith("[") else []

    yes_price = float(prices[0]) if len(prices) > 0 else 0
    no_price = float(prices[1]) if len(prices) > 1 else 0
    pair_cost = yes_price + no_price
    volume = float(market.get("volume", 0) or 0)
    liquidity = float(market.get("liquidity", 0) or 0)
    end_date = market.get("endDate", "")

    # Build analysis
    lines = [
        f"## {question}\n",
        f"**Prices:** YES ${yes_price:.3f} ({yes_price*100:.1f}%) | NO ${no_price:.3f} ({no_price*100:.1f}%)",
        f"**Pair Cost:** ${pair_cost:.4f}",
        f"**Volume:** ${volume:,.0f}",
        f"**Liquidity:** ${liquidity:,.0f}",
    ]

    if end_date:
        lines.append(f"**Resolution Date:** {end_date}")

    # Signals
    lines.append("\n### Signals")

    if pair_cost < 0.98:
        profit = (1 - pair_cost) * 100
        lines.append(f"- **ARBITRAGE DETECTED:** Pair cost ${pair_cost:.4f} = {profit:.2f}% guaranteed profit")
    elif pair_cost < 0.995:
        lines.append(f"- Near-arbitrage: Pair cost ${pair_cost:.4f}")
    else:
        lines.append(f"- No arbitrage (pair cost ${pair_cost:.4f})")

    if volume < 10000:
        lines.append("- Low liquidity warning (<$10k volume)")
    elif volume > 100000:
        lines.append(f"- High liquidity market (${volume:,.0f})")

    if yes_price > 0.9:
        lines.append(f"- Strong YES consensus ({yes_price*100:.1f}%)")
    elif no_price > 0.9:
        lines.append(f"- Strong NO consensus ({no_price*100:.1f}%)")
    elif 0.4 < yes_price < 0.6:
        lines.append("- Market is uncertain (near 50/50)")

    return "\n".join(lines)


def find_arbitrage(min_profit: float = 0.5) -> str:
    """
    Scan for arbitrage opportunities (YES + NO < $1).

    Args:
        min_profit: Minimum profit % to show (default 0.5%)

    Returns:
        List of arbitrage opportunities
    """
    data = _fetch(f"{GAMMA_API}/markets?active=true&closed=false&limit=200")
    if not data:
        return "Failed to fetch markets."

    opportunities = []

    for m in data:
        prices = m.get("outcomePrices", [])
        if isinstance(prices, str):
            prices = json.loads(prices) if prices.startswith("[") else []

        if len(prices) < 2:
            continue

        yes_price = float(prices[0])
        no_price = float(prices[1])
        pair_cost = yes_price + no_price

        if pair_cost < (1 - min_profit/100) and pair_cost > 0.5:
            profit_pct = (1 - pair_cost) * 100
            opportunities.append({
                "question": m.get("question", "Unknown"),
                "yes": yes_price,
                "no": no_price,
                "pair_cost": pair_cost,
                "profit_pct": profit_pct,
                "volume": float(m.get("volume", 0) or 0)
            })

    opportunities = sorted(opportunities, key=lambda x: x["profit_pct"], reverse=True)[:10]

    if not opportunities:
        return f"No arbitrage opportunities found with >{min_profit}% profit."

    lines = ["**Polymarket Arbitrage Opportunities**\n"]
    for i, o in enumerate(opportunities, 1):
        lines.append(
            f"{i}. **{o['question'][:50]}**\n"
            f"   YES ${o['yes']:.3f} + NO ${o['no']:.3f} = ${o['pair_cost']:.4f}\n"
            f"   **Profit: {o['profit_pct']:.2f}%** | Volume: ${o['volume']:,.0f}"
        )

    return "\n".join(lines)


def get_user_positions(wallet_address: str) -> str:
    """
    Get a user's Polymarket positions.

    Args:
        wallet_address: Ethereum wallet address (0x...)

    Returns:
        User's positions and P&L
    """
    wallet = wallet_address.lower()
    if not wallet.startswith("0x"):
        return "Invalid wallet address. Must start with 0x"

    data = _fetch(f"{DATA_API}/positions?user={wallet}")
    if not data:
        return f"No positions found for {wallet[:10]}...{wallet[-6:]}"

    if not isinstance(data, list) or len(data) == 0:
        return f"No open positions for {wallet[:10]}...{wallet[-6:]}"

    lines = [f"**Positions for {wallet[:10]}...{wallet[-6:]}**\n"]
    total_pnl = 0

    for p in data[:20]:
        market = str(p.get("title", p.get("market", "Unknown")))[:40]
        outcome = p.get("outcome", "?")
        size = float(p.get("size", 0))
        avg_price = float(p.get("avgPrice", 0))
        cur_price = float(p.get("curPrice", p.get("currentPrice", 0)))
        pnl = float(p.get("pnl", (cur_price - avg_price) * size))
        total_pnl += pnl

        pnl_str = f"${pnl:+.2f}" if pnl != 0 else "$0.00"
        lines.append(f"- **{market}** ({outcome}): {size:.1f} shares @ ${avg_price:.3f} | Now ${cur_price:.3f} | {pnl_str}")

    lines.append(f"\n**Total Unrealized P&L:** ${total_pnl:+.2f}")

    return "\n".join(lines)
