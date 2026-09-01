"""
Rate limiting configuration.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

from knowledge_transfer_agent.config import get_settings


def get_limiter_key(request):
    """Use client IP for rate limiting."""
    return get_remote_address(request)


def get_limiter() -> Limiter:
    """Create limiter from config with global application limits."""
    s = get_settings()
    limit_str = f"{s.rate_limit_requests}/{s.rate_limit_period}minute"
    return Limiter(
        key_func=get_limiter_key,
        application_limits=[limit_str],
    )
