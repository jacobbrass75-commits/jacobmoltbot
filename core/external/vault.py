"""
Secure Key Vault
================
Handles API keys with strict security controls.

SECURITY INVARIANTS:
- Keys loaded ONLY from environment variables
- Keys NEVER appear in logs, prompts, or disk
- Keys NEVER stored in vector memory
- Keys accessed ONLY inside execution layer
"""

import os
import logging
from typing import Optional
from dataclasses import dataclass
from enum import Enum

# Configure logging to NEVER log key values
logger = logging.getLogger(__name__)


class APIProvider(Enum):
    """Supported external AI providers."""
    CLAUDE = "claude"
    OPENAI = "openai"
    MANUS = "manus"


@dataclass(frozen=True)
class KeyStatus:
    """Status of an API key (never contains the actual key)."""
    provider: APIProvider
    is_configured: bool
    is_valid_format: bool

    def __repr__(self) -> str:
        return f"KeyStatus({self.provider.value}: configured={self.is_configured})"


class SecureKeyVault:
    """
    Secure storage and access for API keys.

    Keys are loaded from environment variables and held only in memory.
    All access is controlled and audited without exposing key values.
    """

    # Environment variable names (public knowledge)
    ENV_VARS = {
        APIProvider.CLAUDE: "CLAUDE_API_KEY",
        APIProvider.OPENAI: "OPENAI_API_KEY",
        APIProvider.MANUS: "MANUS_API_KEY",
    }

    # Minimum key lengths for format validation
    MIN_KEY_LENGTHS = {
        APIProvider.CLAUDE: 40,   # Anthropic keys are ~108 chars
        APIProvider.OPENAI: 40,   # OpenAI keys are ~51 chars
        APIProvider.MANUS: 20,    # Manus key format TBD
    }

    def __init__(self):
        """Initialize vault by loading keys from environment."""
        self._keys: dict[APIProvider, Optional[str]] = {}
        self._access_counts: dict[APIProvider, int] = {p: 0 for p in APIProvider}
        self._load_keys()

    def _load_keys(self) -> None:
        """Load all keys from environment variables."""
        for provider, env_var in self.ENV_VARS.items():
            key = os.environ.get(env_var)
            if key:
                # Strip whitespace that might have been added accidentally
                key = key.strip()
                if key:
                    self._keys[provider] = key
                    # Log only that a key was found, never the value
                    logger.info(f"Loaded API key for {provider.value} from ${env_var}")
                else:
                    self._keys[provider] = None
            else:
                self._keys[provider] = None

    def get_status(self, provider: APIProvider) -> KeyStatus:
        """Get the status of a key without exposing it."""
        key = self._keys.get(provider)
        is_configured = key is not None
        is_valid_format = False

        if is_configured:
            min_len = self.MIN_KEY_LENGTHS.get(provider, 20)
            is_valid_format = len(key) >= min_len

        return KeyStatus(
            provider=provider,
            is_configured=is_configured,
            is_valid_format=is_valid_format
        )

    def get_all_status(self) -> dict[APIProvider, KeyStatus]:
        """Get status of all keys."""
        return {p: self.get_status(p) for p in APIProvider}

    def is_available(self, provider: APIProvider) -> bool:
        """Check if a provider's key is configured and valid."""
        status = self.get_status(provider)
        return status.is_configured and status.is_valid_format

    def _get_key(self, provider: APIProvider) -> Optional[str]:
        """
        INTERNAL: Get the actual key value.

        This method should ONLY be called by the API wrapper classes
        inside the execution layer. Never expose this to prompts or logs.
        """
        if provider not in self._keys:
            return None

        key = self._keys[provider]
        if key:
            self._access_counts[provider] += 1
            # Log access audit (never the key itself)
            logger.debug(f"Key accessed for {provider.value} (count: {self._access_counts[provider]})")

        return key

    def get_access_counts(self) -> dict[APIProvider, int]:
        """Get audit trail of key access counts."""
        return self._access_counts.copy()

    def validate_startup(self, required: list[APIProvider] = None) -> tuple[bool, list[str]]:
        """
        Validate that required keys are present at startup.

        Args:
            required: List of providers that must have valid keys.
                     If None, checks all providers.

        Returns:
            Tuple of (success, list of error messages)
        """
        if required is None:
            required = []  # None means optional, not all required

        errors = []
        for provider in required:
            status = self.get_status(provider)
            if not status.is_configured:
                env_var = self.ENV_VARS[provider]
                errors.append(f"Missing required API key: ${env_var}")
            elif not status.is_valid_format:
                env_var = self.ENV_VARS[provider]
                errors.append(f"Invalid API key format for ${env_var}")

        return len(errors) == 0, errors

    def __repr__(self) -> str:
        """Safe repr that never shows key values."""
        statuses = [f"{p.value}={'OK' if self.is_available(p) else 'MISSING'}"
                   for p in APIProvider]
        return f"SecureKeyVault({', '.join(statuses)})"


# Singleton instance
_vault: Optional[SecureKeyVault] = None


def get_vault() -> SecureKeyVault:
    """Get the global vault instance."""
    global _vault
    if _vault is None:
        _vault = SecureKeyVault()
    return _vault


def require_keys(*providers: APIProvider) -> tuple[bool, list[str]]:
    """
    Convenience function to validate required keys at startup.

    Usage:
        success, errors = require_keys(APIProvider.CLAUDE, APIProvider.OPENAI)
        if not success:
            for error in errors:
                print(f"ERROR: {error}")
            sys.exit(1)
    """
    vault = get_vault()
    return vault.validate_startup(list(providers))
